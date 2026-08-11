#!/usr/bin/env python3
"""Microbenchmark default-off VideoVAE decoder QuantizedLinear loading/decode.

The parent mode runs fresh child processes for unquantized full-grid decode and one quantized
variant so load time, decode time, MLX peak, RSS and VM deltas are not polluted by prior variants.
It uses only the local ``models/MiniMax-H3/FL2VA/video_vae/source/model.safetensors`` weights.
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
    (out_dir / f"{label}_vm_stat.txt").write_text(vm.stdout + (("\nSTDERR:\n" + vm.stderr) if vm.stderr else ""))
    (out_dir / f"{label}_swapusage.txt").write_text(swap.stdout + (("\nSTDERR:\n" + swap.stderr) if swap.stderr else ""))
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
            func = getattr(mx, name, None) or getattr(metal, name, None) if metal is not None else getattr(mx, name, None)
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

    metal = getattr(mx, "metal", None)
    for owner in (mx, metal):
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


def child_main(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from minimax_h3_mlx.load import load_video_vae
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
    vae = load_video_vae(args.video_vae_dir, decoder_quantization=args.decoder_quantization)
    vae.set_decode_spatial_tiling(False)
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

    tensor_path = out_dir / f"decoded_{args.decoder_quantization}.npy"
    np.save(tensor_path, decoded_np)
    pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
    pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
    image = np.clip(decoded_np * pixel_std + pixel_mean, 0.0, 1.0)
    image_uint8 = (image[0].transpose(1, 2, 3, 0) * 255.0 + 0.5).astype(np.uint8)
    image_path = out_dir / f"image_{args.decoder_quantization}.npy"
    np.save(image_path, image_uint8)

    result = {
        "variant": args.decoder_quantization,
        "created_utc": iso_now(),
        "video_vae_dir": args.video_vae_dir,
        "latent_shape": list(latent_shape),
        "decoded_shape": list(decoded_np.shape),
        "image_shape": list(image_uint8.shape),
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "load_plus_decode_seconds": load_seconds + decode_seconds,
        "after_load_memory": after_load_memory,
        "after_decode_memory": after_decode_memory,
        "tensor_path": str(tensor_path),
        "image_path": str(image_path),
        "decoder_quantization_summary": getattr(vae, "video_vae_decoder_quantization", None),
        "resource_usage_self": {"ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    path = out_dir / f"child_{args.decoder_quantization}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


def run_child(parent_args: argparse.Namespace, variant: str, run_dir: Path) -> dict[str, Any]:
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    before = memory_sample(f"pre_{variant}", run_dir)
    stdout = variant_dir / "child.stdout"
    stderr = variant_dir / "child.stderr"
    cmd = [
        "/usr/bin/time",
        "-l",
        str(ROOT / ".venv/bin/python"),
        str(Path(__file__).resolve()),
        "--child",
        "--decoder-quantization",
        variant,
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
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=parent_args.timeout_seconds)
    elapsed = time.perf_counter() - started
    stdout.write_text(proc.stdout)
    stderr.write_text(proc.stderr)
    after = memory_sample(f"post_{variant}", run_dir)
    payload: dict[str, Any] = {}
    child_json = variant_dir / f"child_{variant}.json"
    if child_json.exists():
        payload = json.loads(child_json.read_text())
    return {
        "variant": variant,
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
    return {"shape": list(base.shape), "max_abs": max_abs, "mean_abs": mean_abs, "rmse": rmse, "psnr_db_relative_to_base_abs_peak": psnr}


def quantized_linear_semantics_probe(run_dir: Path) -> dict[str, Any]:
    """Record source-local MLX QuantizedLinear save/load behavior used by the loader."""

    import inspect
    import tempfile

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=True)

        def __call__(self, x):
            return self.proj(x)

    mx.random.seed(123)
    model = Tiny()
    x = mx.random.normal((3, 64)).astype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear))
    y = model(x)
    mx.eval(y)
    params = dict(tree_flatten(model.parameters()))
    with tempfile.TemporaryDirectory(dir=run_dir) as tmp:
        path = Path(tmp) / "tiny_quantized.safetensors"
        mx.save_safetensors(str(path), params)
        reloaded = Tiny()
        nn.quantize(reloaded, group_size=64, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear))
        reloaded.update(tree_unflatten(list(mx.load(str(path)).items())))
        y2 = reloaded(x)
        mx.eval(y2)
    return {
        "mlx_nn_quantize_signature": str(inspect.signature(nn.quantize)),
        "quantized_linear_class": f"{nn.QuantizedLinear.__module__}.{nn.QuantizedLinear.__name__}",
        "quantized_linear_source_calls_quantized_matmul": "quantized_matmul" in inspect.getsource(nn.QuantizedLinear),
        "parameter_keys": sorted(params),
        "weight_dtype": str(params["proj.weight"].dtype),
        "scales_key_present": "proj.scales" in params,
        "biases_key_present": "proj.biases" in params,
        "bias_key_present": "proj.bias" in params,
        "bits": int(model.proj.bits),
        "group_size": int(model.proj.group_size),
        "reload_max_abs_delta": float(mx.max(mx.abs(y2 - y)).item()),
        "verdict": "QuantizedLinear parameters are packed weight/scales/biases plus optional bias; save/load is exact when the module tree is quantized before update.",
    }


def parent_main(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "experiments" / f"video_vae_decoder_quant_microbench_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    variants = ["off", args.decoder_quantization]
    if args.include_4bit and "4bit" not in variants:
        variants.append("4bit")
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": iso_now(),
        "objective": "focused real VideoVAE 320x192-shaped full-grid decode microbench for decoder QuantizedLinear loading",
        "run_dir": str(run_dir),
        "environment": {"python": sys.version, "platform": platform.platform()},
        "mlx_quantized_linear_semantics": quantized_linear_semantics_probe(run_dir),
        "inputs": {
            "video_vae_dir": args.video_vae_dir,
            "source_weights": str(Path(args.video_vae_dir) / "source" / "model.safetensors"),
            "width": args.width,
            "height": args.height,
            "duration_seconds": args.duration,
            "seed": args.seed,
            "variants": variants,
            "full_grid_decode": True,
            "decode_internal_eval_boundaries": True,
        },
        "variants": [],
        "comparisons": {},
    }
    rc = 0
    for variant in variants:
        try:
            row = run_child(args, variant, run_dir)
        except subprocess.TimeoutExpired as exc:
            row = {"variant": variant, "returncode": 124, "timeout_seconds": args.timeout_seconds, "error": str(exc)}
        result["variants"].append(row)
        if row.get("returncode") != 0:
            rc = 1
            break
    by_variant = {row.get("variant"): row for row in result["variants"]}
    baseline = by_variant.get("off", {}).get("child_result", {})
    for variant in variants:
        if variant == "off":
            continue
        cand = by_variant.get(variant, {}).get("child_result", {})
        if baseline.get("tensor_path") and cand.get("tensor_path"):
            result["comparisons"][f"off_vs_{variant}_tensor"] = compare_arrays(baseline["tensor_path"], cand["tensor_path"])
        if baseline.get("image_path") and cand.get("image_path"):
            result["comparisons"][f"off_vs_{variant}_image_uint8"] = compare_arrays(baseline["image_path"], cand["image_path"])
    # Conservative decision hint; final keep/reject is written by the operator-facing result note.
    if rc == 0 and args.decoder_quantization in by_variant:
        base_ld = float(baseline.get("load_plus_decode_seconds") or 0.0)
        cand_ld = float(by_variant[args.decoder_quantization].get("child_result", {}).get("load_plus_decode_seconds") or 0.0)
        result["decision_hint"] = "microbench_speed_positive" if cand_ld and base_ld and cand_ld < base_ld else "microbench_not_speed_positive"
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "result": str(run_dir / "result.json"), "status": "ok" if rc == 0 else "failed"}, indent=2))
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--video-vae-dir", default="models/MiniMax-H3/FL2VA/video_vae")
    parser.add_argument("--decoder-quantization", choices=("off", "8bit", "4bit"), default="8bit")
    parser.add_argument("--include-4bit", action="store_true")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--memory-limit-gb", type=float, default=16.0)
    parser.add_argument("--timeout-seconds", type=int, default=360)
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
