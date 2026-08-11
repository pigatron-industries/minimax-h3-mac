#!/usr/bin/env python3
"""Microbenchmark default-off VideoVAE lower-precision loading/decode.

The parent process launches each variant in a fresh child so load time, decode time, MLX memory,
RSS, and VM deltas are not polluted by prior variants. Variants cover historical FP32 parameter
loading, BF16 loading/decode, and FP16 loading/decode under both default tiled and full-grid decode.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_vm_stat(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    match = re.search(r"page size of (\d+) bytes", text)
    if match:
        out["page_size_bytes"] = int(match.group(1))
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        nums = re.findall(r"\d+", raw.replace(".", ""))
        if nums:
            out[key.strip().replace(" ", "_").replace('"', "")] = int(nums[0])
    return out


def parse_swapusage(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"raw": text.strip()}
    for key in ("total", "used", "free"):
        match = re.search(rf"{key}\s*=\s*([0-9.]+)M", text)
        if match:
            out[f"{key}_mb"] = float(match.group(1))
    return out


def memory_sample(label: str, out_dir: Path) -> dict[str, Any]:
    vm = subprocess.run(["vm_stat"], text=True, capture_output=True, timeout=10)
    swap = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, timeout=10)
    ps = subprocess.run(["ps", "axo", "pid,ppid,rss,command"], text=True, capture_output=True, timeout=10)
    (out_dir / f"{label}_vm_stat.txt").write_text(vm.stdout + (("\nSTDERR:\n" + vm.stderr) if vm.stderr else ""))
    (out_dir / f"{label}_swapusage.txt").write_text(swap.stdout + (("\nSTDERR:\n" + swap.stderr) if swap.stderr else ""))
    (out_dir / f"{label}_processes.txt").write_text(ps.stdout + (("\nSTDERR:\n" + ps.stderr) if ps.stderr else ""))
    return {
        "label": label,
        "created_utc": iso_now(),
        "vm": parse_vm_stat(vm.stdout),
        "swapusage": parse_swapusage(swap.stdout),
    }


def memory_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    bvm = before.get("vm", {})
    avm = after.get("vm", {})
    page_size = avm.get("page_size_bytes") or bvm.get("page_size_bytes")
    out: dict[str, Any] = {"page_size_bytes": page_size}
    for key in ("Pageouts", "Swapouts", "Pageins", "Swapins", "Compressions", "Decompressions"):
        if key in avm and key in bvm:
            out[key.lower()] = avm[key] - bvm[key]
    if page_size:
        for key in ("pageouts", "swapouts", "pageins", "swapins"):
            if key in out:
                out[f"{key}_bytes"] = out[key] * page_size
    bsw = before.get("swapusage", {})
    asw = after.get("swapusage", {})
    for key in ("total_mb", "used_mb", "free_mb"):
        if key in asw and key in bsw:
            out[f"swapusage_{key}_delta"] = asw[key] - bsw[key]
    return out


def parse_time_l(stderr: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    patterns = {
        "max_rss_bytes": r"(\d+)\s+maximum resident set size",
        "page_reclaims": r"(\d+)\s+page reclaims",
        "page_faults": r"(\d+)\s+page faults",
        "process_swaps": r"(\d+)\s+swaps",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr)
        if match:
            out[key] = int(match.group(1))
    return out


def mlx_memory() -> dict[str, int | None]:
    import mlx.core as mx

    metal = getattr(mx, "metal", None)
    out: dict[str, int | None] = {}
    for key, names in {
        "active_bytes": ("get_active_memory",),
        "cache_bytes": ("get_cache_memory",),
        "peak_bytes": ("get_peak_memory", "get_peak_memory_usage"),
    }.items():
        value = None
        for name in names:
            func = getattr(mx, name, None)
            if func is None and metal is not None:
                func = getattr(metal, name, None)
            if func is None:
                continue
            try:
                value = int(func())
                break
            except Exception:
                pass
        out[key] = value
    return out


def reset_mlx_peak() -> None:
    import mlx.core as mx

    for owner in (mx, getattr(mx, "metal", None)):
        if owner is None:
            continue
        for name in ("reset_peak_memory", "reset_peak_memory_stats"):
            func = getattr(owner, name, None)
            if func is not None:
                try:
                    func()
                    return
                except Exception:
                    pass


def configure_memory_guard(memory_limit_gb: float) -> None:
    import mlx.core as mx

    limit = int(memory_limit_gb * 1e9)
    for owner in (mx, getattr(mx, "metal", None)):
        if owner is None:
            continue
        for name, args in (
            ("set_cache_limit", (0,)),
            ("set_memory_limit", (limit,)),
            ("set_wired_limit", (limit,)),
        ):
            func = getattr(owner, name, None)
            if func is not None:
                try:
                    func(*args)
                except Exception:
                    pass


def variant_name(precision: str, full_grid: bool) -> str:
    return f"{precision}_{'full_grid' if full_grid else 'default_tiled'}"


def child_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from minimax_h3_mlx.load import load_video_vae, video_vae_parameter_dtype_summary
    from minimax_h3_mlx.packing import FPS, PIXEL_MEAN, PIXEL_STD, align_num_frames, video_latent_num_frames

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_memory_guard(args.memory_limit_gb)
    reset_mlx_peak()
    mx.random.seed(args.seed)

    num_frames = align_num_frames(int(round(args.duration * FPS)))
    num_latent_frames = video_latent_num_frames(num_frames)
    latent_shape = (1, 24, num_latent_frames, args.height // 16, args.width // 16)
    rng = np.random.default_rng(args.seed)
    latents = mx.array(rng.standard_normal(latent_shape).astype(np.float32))
    mx.eval(latents)

    load_started = time.perf_counter()
    vae = load_video_vae(args.video_vae_dir, precision=args.precision)
    vae.set_decode_spatial_tiling(not args.full_grid)
    vae.set_decode_internal_eval_boundaries(True)
    mx.eval(vae.parameters())
    load_seconds = time.perf_counter() - load_started
    after_load_memory = mlx_memory()

    reset_mlx_peak()
    decode_started = time.perf_counter()
    decoded = vae.decode(latents)
    mx.eval(decoded)
    decoded_np = np.array(decoded, dtype=np.float32)
    decode_seconds = time.perf_counter() - decode_started
    after_decode_memory = mlx_memory()

    name = variant_name(args.precision, args.full_grid)
    tensor_path = out_dir / f"decoded_{name}.npy"
    np.save(tensor_path, decoded_np)
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    image = np.clip(decoded_np * pixel_std + pixel_mean, 0.0, 1.0)
    image_uint8 = (image[0].transpose(1, 2, 3, 0) * 255.0 + 0.5).astype(np.uint8)
    image_path = out_dir / f"image_{name}.npy"
    np.save(image_path, image_uint8)

    result = {
        "variant": name,
        "created_utc": iso_now(),
        "video_vae_dir": args.video_vae_dir,
        "precision": args.precision,
        "full_grid_decode": bool(args.full_grid),
        "default_tiled_decode": not bool(args.full_grid),
        "latent_shape": list(latent_shape),
        "decoded_shape": list(decoded_np.shape),
        "decoded_dtype_after_numpy": str(decoded_np.dtype),
        "image_shape": list(image_uint8.shape),
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "load_plus_decode_seconds": load_seconds + decode_seconds,
        "after_load_memory": after_load_memory,
        "after_decode_memory": after_decode_memory,
        "tensor_path": str(tensor_path),
        "image_path": str(image_path),
        "video_vae_precision_summary": getattr(vae, "video_vae_precision", None),
        "video_vae_parameter_summary": getattr(vae, "video_vae_parameter_summary", video_vae_parameter_dtype_summary(vae)),
        "resource_usage_self": {"ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    path = out_dir / f"child_{name}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


def run_child(parent_args: argparse.Namespace, precision: str, full_grid: bool, run_dir: Path) -> dict[str, Any]:
    name = variant_name(precision, full_grid)
    variant_dir = run_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    before = memory_sample(f"pre_{name}", run_dir)
    stdout = variant_dir / "child.stdout"
    stderr = variant_dir / "child.stderr"
    cmd = [
        "/usr/bin/time",
        "-l",
        str(ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--child",
        "--precision",
        precision,
        "--video-vae-dir",
        parent_args.video_vae_dir,
        "--width",
        str(parent_args.width),
        "--height",
        str(parent_args.height),
        "--duration",
        str(parent_args.duration),
        "--seed",
        str(parent_args.seed),
        "--memory-limit-gb",
        str(parent_args.memory_limit_gb),
        "--out-dir",
        str(variant_dir),
    ]
    if full_grid:
        cmd.append("--full-grid")
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=parent_args.timeout_seconds)
    elapsed = time.perf_counter() - started
    stdout.write_text(proc.stdout)
    stderr.write_text(proc.stderr)
    after = memory_sample(f"post_{name}", run_dir)
    payload: dict[str, Any] = {}
    child_json = variant_dir / f"child_{name}.json"
    if child_json.exists():
        payload = json.loads(child_json.read_text())
    return {
        "variant": name,
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_wall_seconds_python": elapsed,
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
        "time_l_metrics": parse_time_l(proc.stderr),
        "memory": {"pre": before, "post": after, "delta": memory_delta(before, after)},
        "child_result": payload,
    }


def compare_arrays(base_path: str, cand_path: str) -> dict[str, Any]:
    base = np.load(base_path)
    cand = np.load(cand_path)
    diff = cand.astype(np.float32) - base.astype(np.float32)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    peak = float(np.max(np.abs(base.astype(np.float32))))
    psnr = float("inf") if rmse == 0 else float(20.0 * np.log10(max(peak, 1e-12) / rmse))
    return {
        "shape": list(base.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "psnr_db_relative_to_base_abs_peak": psnr,
    }


def parent_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "experiments" / f"video_vae_precision_microbench_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    variants = [(precision, full_grid) for full_grid in (False, True) for precision in ("fp32", "bf16", "fp16")]
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": iso_now(),
        "objective": "focused real VideoVAE 320x192-shaped decode microbench for BF16/FP16 parameter/decode precision",
        "run_dir": str(run_dir),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "inputs": {
            "video_vae_dir": args.video_vae_dir,
            "source_weights": str(Path(args.video_vae_dir) / "source" / "model.safetensors"),
            "width": args.width,
            "height": args.height,
            "duration_seconds": args.duration,
            "seed": args.seed,
            "variants": [variant_name(p, fg) for p, fg in variants],
            "decode_internal_eval_boundaries": True,
        },
        "variants": [],
        "comparisons": {},
    }
    rc = 0
    for precision, full_grid in variants:
        try:
            row = run_child(args, precision, full_grid, run_dir)
        except subprocess.TimeoutExpired as exc:
            row = {"variant": variant_name(precision, full_grid), "returncode": 124, "timeout_seconds": args.timeout_seconds, "error": str(exc)}
        result["variants"].append(row)
        if row.get("returncode") != 0:
            rc = 1
            break
    by_variant = {row.get("variant"): row for row in result["variants"]}
    comparisons = [
        ("fp32_default_tiled", "bf16_default_tiled"),
        ("fp32_default_tiled", "fp16_default_tiled"),
        ("fp32_full_grid", "bf16_full_grid"),
        ("fp32_full_grid", "fp16_full_grid"),
        ("bf16_default_tiled", "fp16_default_tiled"),
        ("bf16_full_grid", "fp16_full_grid"),
        ("fp32_default_tiled", "fp32_full_grid"),
        ("bf16_default_tiled", "bf16_full_grid"),
        ("fp16_default_tiled", "fp16_full_grid"),
    ]
    for base_name, cand_name in comparisons:
        base = by_variant.get(base_name, {}).get("child_result", {})
        cand = by_variant.get(cand_name, {}).get("child_result", {})
        if base.get("tensor_path") and cand.get("tensor_path"):
            result["comparisons"][f"{base_name}_vs_{cand_name}_tensor"] = compare_arrays(base["tensor_path"], cand["tensor_path"])
        if base.get("image_path") and cand.get("image_path"):
            result["comparisons"][f"{base_name}_vs_{cand_name}_image_uint8"] = compare_arrays(base["image_path"], cand["image_path"])

    def child_seconds(name: str, key: str) -> float | None:
        value = by_variant.get(name, {}).get("child_result", {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None

    tiled_base = child_seconds("fp32_default_tiled", "load_plus_decode_seconds")
    tiled_bf16 = child_seconds("bf16_default_tiled", "load_plus_decode_seconds")
    tiled_fp16 = child_seconds("fp16_default_tiled", "load_plus_decode_seconds")
    full_base = child_seconds("fp32_full_grid", "load_plus_decode_seconds")
    full_bf16 = child_seconds("bf16_full_grid", "load_plus_decode_seconds")
    full_fp16 = child_seconds("fp16_full_grid", "load_plus_decode_seconds")

    def speedup_percent(base: float | None, cand: float | None) -> float | None:
        return 100.0 * (base - cand) / base if base and cand else None

    def signal(base: float | None, cand: float | None) -> str:
        return "positive" if base and cand and cand < base else "not_positive"

    fp16_candidates = {
        "fp16_default_tiled": tiled_fp16,
        "fp16_full_grid": full_fp16,
    }
    viable_fp16 = {name: seconds for name, seconds in fp16_candidates.items() if seconds is not None and seconds > 0}
    best_fp16_name = min(viable_fp16, key=viable_fp16.get) if viable_fp16 else None
    result["decision_hint"] = {
        "bf16_default_tiled_load_plus_decode_speedup_percent": speedup_percent(tiled_base, tiled_bf16),
        "fp16_default_tiled_load_plus_decode_speedup_percent": speedup_percent(tiled_base, tiled_fp16),
        "bf16_full_grid_load_plus_decode_speedup_percent": speedup_percent(full_base, full_bf16),
        "fp16_full_grid_load_plus_decode_speedup_percent": speedup_percent(full_base, full_fp16),
        "bf16_default_tiled_signal": signal(tiled_base, tiled_bf16),
        "fp16_default_tiled_signal": signal(tiled_base, tiled_fp16),
        "bf16_full_grid_signal": signal(full_base, full_bf16),
        "fp16_full_grid_signal": signal(full_base, full_fp16),
        "best_fp16_variant_by_load_plus_decode_seconds": best_fp16_name,
        "best_fp16_load_plus_decode_seconds": viable_fp16.get(best_fp16_name) if best_fp16_name else None,
        "fp16_microbench_signal": "positive" if (
            (tiled_base and tiled_fp16 and tiled_fp16 < tiled_base)
            or (full_base and full_fp16 and full_fp16 < full_base)
        ) else "not_positive",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "result": str(run_dir / "result.json"), "status": "ok" if rc == 0 else "failed"}, indent=2))
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--video-vae-dir", default="models/MiniMax-H3/FL2VA/video_vae")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    if args.child:
        if args.out_dir is None:
            parser.error("--child requires --out-dir")
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
