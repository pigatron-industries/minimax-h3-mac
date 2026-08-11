#!/usr/bin/env python3
"""Run one real MiniMax-H3 generation with opt-in forward profiling.

This is intentionally a bounded experiment runner, not a release pipeline.  It
creates experiments/forward_pass_profile_<UTC>/ with raw command logs and a
result.json that combines the profiled generation, memory deltas, media validity,
and an MLX activation-quantization support probe.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_capture(cmd: list[str], *, stdout: Path, stderr: Path, timeout: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    stdout.write_text(proc.stdout)
    stderr.write_text(proc.stderr)
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_wall_seconds_python": time.perf_counter() - started,
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
    }


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


def is_h3_mlx_process(line: str) -> bool:
    lower = line.lower()
    if any(
        token in lower
        for token in (
            "scripts/generate.py",
            "run_with_mlx_memory.py",
            "run_forward_profile_experiment.py",
        )
    ):
        return True
    return "python" in lower and ("minimax-h3" in lower or "minimax_h3" in lower)


def resolution_requires_healthy_preflight(resolution: str) -> bool:
    width_s, height_s = resolution.lower().split("x", 1)
    return int(width_s) * int(height_s) >= 320 * 192


def memory_sample(label: str, memory_dir: Path) -> dict[str, Any]:
    vm = subprocess.run(["vm_stat"], text=True, capture_output=True, timeout=10)
    swap = subprocess.run(["sysctl", "vm.swapusage"], text=True, capture_output=True, timeout=10)
    ps = subprocess.run(["ps", "axo", "pid,ppid,rss,command"], text=True, capture_output=True, timeout=10)
    vm_path = memory_dir / f"{label}_vm_stat.txt"
    swap_path = memory_dir / f"{label}_swapusage.txt"
    ps_path = memory_dir / f"{label}_processes.txt"
    vm_path.write_text(vm.stdout + (("\nSTDERR:\n" + vm.stderr) if vm.stderr else ""))
    swap_path.write_text(swap.stdout + (("\nSTDERR:\n" + swap.stderr) if swap.stderr else ""))
    ps_path.write_text(ps.stdout + (("\nSTDERR:\n" + ps.stderr) if ps.stderr else ""))
    matches = []
    for line in ps.stdout.splitlines()[1:]:
        if is_h3_mlx_process(line):
            if str(os.getpid()) not in line.split(None, 1)[0:1]:
                matches.append(line.rstrip())
    return {
        "label": label,
        "created_utc": iso_now(),
        "vm_stat_returncode": vm.returncode,
        "swapusage_returncode": swap.returncode,
        "process_returncode": ps.returncode,
        "vm": parse_vm_stat(vm.stdout),
        "swapusage": parse_swapusage(swap.stdout),
        "matching_h3_mlx_processes": matches,
        "raw_paths": {"vm_stat": str(vm_path), "swapusage": str(swap_path), "processes": str(ps_path)},
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
        if "pageouts" in out:
            out["pageouts_bytes"] = out["pageouts"] * page_size
        if "swapouts" in out:
            out["swapouts_bytes"] = out["swapouts"] * page_size
    bsw = before.get("swapusage", {})
    asw = after.get("swapusage", {})
    for key in ("total_mb", "used_mb", "free_mb"):
        if key in asw and key in bsw:
            out[f"swapusage_{key}_delta"] = asw[key] - bsw[key]
    return out


def preflight(memory_dir: Path, *, idle_seconds: float) -> dict[str, Any]:
    first = memory_sample("preflight_idle_1", memory_dir)
    time.sleep(float(idle_seconds))
    second = memory_sample("preflight_idle_2", memory_dir)
    delta = memory_delta(first, second)
    swap_free = second.get("swapusage", {}).get("free_mb")
    raw_matches = first["matching_h3_mlx_processes"] + second["matching_h3_mlx_processes"]
    real_matches = [line for line in raw_matches if "run_forward_profile_experiment.py" not in line]
    issues = []
    risk_signals = []
    if delta.get("pageouts") not in (None, 0):
        issues.append(f"idle_pageouts_delta_nonzero:{delta.get('pageouts')}")
    if delta.get("swapouts") not in (None, 0):
        issues.append(f"idle_swapouts_delta_nonzero:{delta.get('swapouts')}")
    if swap_free is None:
        risk_signals.append("swapusage_unparsed")
    elif swap_free < 2048.0:
        risk_signals.append(f"low_swap_free_risk_below_2048_mb:{swap_free}")
    if real_matches:
        issues.append("active_h3_mlx_process_matches_present")
    return {
        "samples": [first, second],
        "idle_interval_seconds": idle_seconds,
        "delta": delta,
        "swapusage_after_mb": second.get("swapusage", {}),
        "health_criteria": {
            "idle_pageouts_delta_required": 0,
            "idle_swapouts_delta_required": 0,
            "low_swap_free_mb_recorded_as_risk_not_hard_gate": 2048.0,
            "no_active_h3_mlx_process_matches": True,
        },
        "issues": issues,
        "risk_signals": risk_signals,
        "healthy": not issues,
        "real_active_h3_mlx_process_matches": real_matches,
    }


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
    mlx = re.search(r"ARGUS_MLX_MEMORY peak_bytes=(\S+) active_bytes=(\S+) cache_bytes=(\S+)", stderr)
    if mlx:
        for key, raw in zip(("mlx_peak_bytes", "mlx_active_bytes_at_exit", "mlx_cache_bytes_at_exit"), mlx.groups()):
            out[key] = None if raw == "None" else int(raw)
    return out


def parse_generation_stdout(stdout: str) -> dict[str, Any]:
    """Extract lightweight generation stats printed by scripts/generate.py."""

    out: dict[str, Any] = {}
    step = re.search(r"([0-9.]+)s per step, ([0-9.]+) min total", stdout)
    if step:
        out["pipeline_seconds_per_step"] = float(step.group(1))
        out["pipeline_total_minutes_printed"] = float(step.group(2))
    cache = re.search(
        r"block cache:\s+full=(\d+)\s+cached=(\d+)\s+skipped=(\d+)\s+blocks\s+\(([0-9.]+)% of block executions\)",
        stdout,
    )
    if cache:
        out["block_cache"] = {
            "full_steps": int(cache.group(1)),
            "cache_steps": int(cache.group(2)),
            "skipped_blocks": int(cache.group(3)),
            "saved_fraction_percent": float(cache.group(4)),
        }
    return out


def media_validation(path: Path, *, ffmpeg: Path, ffprobe: Path, logs: Path) -> dict[str, Any]:
    from minimax_h3_mlx.media_metrics import audio_activity_metrics, decode_audio_mono_f32, decode_video_rgb, probe_media

    payload: dict[str, Any] = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else None}
    try:
        info = probe_media(path, ffprobe=ffprobe)
        ffprobe_path = logs / "ffprobe_profiled_generation.json"
        ffprobe_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        video, video_meta = decode_video_rgb(path, ffmpeg=ffmpeg, ffprobe=ffprobe, max_frames=1)
        audio = decode_audio_mono_f32(path, ffmpeg=ffmpeg)
        activity = audio_activity_metrics(audio)
        payload.update(
            {
                "ffprobe_path": str(ffprobe_path),
                "video": {
                    **video_meta,
                    "first_frame_shape": list(video.shape),
                    "first_frame_min": int(video.min()),
                    "first_frame_max": int(video.max()),
                },
                "audio": activity,
                "valid": bool(activity.get("activity_ok") and video.shape[0] > 0),
                "error": None,
            }
        )
    except Exception as exc:
        payload.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
    return payload


def command_versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": sys.version}
    try:
        import mlx
        import mlx.core as mx

        out["mlx_version"] = getattr(mlx, "__version__", None)
        if out["mlx_version"] is None:
            try:
                out["mlx_version"] = importlib.metadata.version("mlx")
            except importlib.metadata.PackageNotFoundError:
                out["mlx_version"] = None
        try:
            out["mlx_metal_version"] = importlib.metadata.version("mlx-metal")
        except importlib.metadata.PackageNotFoundError:
            out["mlx_metal_version"] = None
        out["mlx_default_device"] = str(mx.default_device())
        out["mlx_metal_available"] = hasattr(mx, "metal")
    except Exception as exc:
        out["mlx_error"] = f"{type(exc).__name__}: {exc}"
    out["platform"] = {
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    return out


def safe_signature(obj: Any) -> str | None:
    import inspect

    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def quantized_activation_probe() -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(123)
    public_symbols = {
        "mlx.core": sorted(name for name in dir(mx) if "quant" in name.lower()),
        "mlx.nn": sorted(name for name in dir(nn) if "quant" in name.lower()),
    }
    signatures = {
        "mx.quantize": safe_signature(getattr(mx, "quantize", None)) if hasattr(mx, "quantize") else None,
        "mx.quantized_matmul": safe_signature(getattr(mx, "quantized_matmul", None)) if hasattr(mx, "quantized_matmul") else None,
        "nn.quantize": safe_signature(getattr(nn, "quantize", None)) if hasattr(nn, "quantize") else None,
        "nn.QuantizedLinear": safe_signature(getattr(nn, "QuantizedLinear", None)) if hasattr(nn, "QuantizedLinear") else None,
    }
    probe_inputs = {
        "tiny_linear": {
            "input_shape": [2, 64],
            "input_dtype": "mlx.core.bfloat16",
            "layer": "nn.Linear(64, 32, bias=False)",
            "group_size": 64,
            "bits": 4,
        },
        "weight_storage_quantization_mode": "affine",
        "activation_quantize_input_modes_tested": ["affine", "mxfp8", "nvfp4"],
    }
    observations: dict[str, Any] = {}
    try:
        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(64, 32, bias=False)

            def __call__(self, value: mx.array) -> mx.array:
                return self.proj(value)

        tiny = Tiny()
        nn.quantize(tiny, group_size=64, bits=4)
        linear = tiny.proj
        x = mx.random.normal((2, 64)).astype(mx.bfloat16)
        y = tiny(x)
        mx.eval(y)
        observations["quantized_linear_smoke"] = {
            "ok": True,
            "class": type(linear).__name__,
            "input_shape": list(x.shape),
            "input_dtype": str(x.dtype),
            "output_dtype": str(y.dtype),
            "weight_dtype": str(getattr(linear, "weight").dtype),
            "weight_shape": list(getattr(linear, "weight").shape),
            "scales_present": hasattr(linear, "scales"),
            "scales_dtype": str(getattr(linear, "scales").dtype) if hasattr(linear, "scales") else None,
            "bits": int(getattr(linear, "bits", 0) or 0),
            "group_size": int(getattr(linear, "group_size", 0) or 0),
        }
    except Exception as exc:
        observations["quantized_linear_smoke"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    activation_mode_smokes = []
    for mode in ("affine", "mxfp8", "nvfp4"):
        row: dict[str, Any] = {
            "mode": mode,
            "quantize_input": True,
            "input_shape": probe_inputs["tiny_linear"]["input_shape"],
            "input_dtype": probe_inputs["tiny_linear"]["input_dtype"],
            "layer": probe_inputs["tiny_linear"]["layer"],
            "group_size": probe_inputs["tiny_linear"]["group_size"],
            "bits": probe_inputs["tiny_linear"]["bits"],
        }
        try:
            tiny = Tiny()
            nn.quantize(tiny, group_size=64, bits=4, mode=mode, quantize_input=True)
            x = mx.random.normal((2, 64)).astype(mx.bfloat16)
            y = tiny(x)
            mx.eval(y)
            row.update(
                {
                    "ok": True,
                    "class": type(tiny.proj).__name__,
                    "input_dtype": str(x.dtype),
                    "output_dtype": str(y.dtype),
                    "weight_dtype": str(tiny.proj.weight.dtype),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        activation_mode_smokes.append(row)
    observations["quantize_input_activation_smokes"] = activation_mode_smokes

    activation_api_candidates = []
    for namespace, names in public_symbols.items():
        for name in names:
            lower = name.lower()
            if "activation" in lower or "act" in lower or "observer" in lower:
                activation_api_candidates.append(f"{namespace}.{name}")
    if any(row.get("ok") for row in activation_mode_smokes):
        verdict = "activation_quantization_supported_for_tested_tiny_linear"
        reason = "nn.quantize(..., quantize_input=True) executed and evaluated for at least one tested mode."
    elif any("NYI" in str(row.get("error", "")) or "not supported" in str(row.get("error", "")).lower() for row in activation_mode_smokes):
        verdict = "activation_quantization_api_present_but_runtime_rejected"
        reason = (
            "Installed MLX exposes nn.quantize(..., quantize_input=True), but affine rejects quantized "
            "activations and the activation-capable mxfp8/nvfp4 modes fail this general Linear smoke "
            "with QQMatmul NYI. Treat activation quantization as rejected-for-now on this macOS/MLX route."
        )
    else:
        verdict = "unsupported_public_weight_only"
        reason = (
            "Installed MLX exposes weight/storage quantization APIs but no working public activation "
            "quantization route was observed in the tested Linear smoke."
        )
    return {
        "created_utc": iso_now(),
        "public_quant_symbols": public_symbols,
        "signatures": signatures,
        "probe_inputs": probe_inputs,
        "observations": observations,
        "activation_api_candidates": activation_api_candidates,
        "verdict": verdict,
        "distinguishes_weight_only_from_activation_quantization": True,
        "reason": reason,
    }


def _total_seconds(container: dict[str, Any], key: str) -> float:
    try:
        return float(container.get(key, {}).get("total_seconds") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _max_observed_bytes(container: dict[str, Any], keys: tuple[str, ...], counter: str) -> int | None:
    values = [
        container.get(key, {}).get(counter)
        for key in keys
        if isinstance(container.get(key), dict)
    ]
    finite = [int(value) for value in values if isinstance(value, (int, float))]
    return max(finite) if finite else None


def build_acceptance_summary(result: dict[str, Any]) -> dict[str, Any]:
    generation = result.get("generation", {})
    profile_summary = generation.get("forward_profile_summary", {}) or {}
    labels = profile_summary.get("label_totals", {}) or {}
    categories = profile_summary.get("category_totals", {}) or {}

    load_init_components = {
        "text_encoder_load": _total_seconds(labels, "load.text_encoder_low_memory"),
        "transformer_static_low_memory_load": _total_seconds(labels, "load.streaming_transformer_low_memory"),
        "video_vae_load": _total_seconds(labels, "load.video_vae_low_memory"),
        "audio_vae_load": _total_seconds(labels, "load.audio_vae_low_memory"),
        "adaln_cache_build": _total_seconds(labels, "pipeline.adaln_cache_build"),
    }
    load_init_total = sum(load_init_components.values())
    lazy_block_load = _total_seconds(labels, "streaming.block_load")
    stage_timing = {
        "text_conditioning": (
            _total_seconds(labels, "load.text_encoder_low_memory")
            + _total_seconds(labels, "pipeline.text_encoder_encode")
        ),
        "dit_setup": (
            _total_seconds(labels, "load.streaming_transformer_low_memory")
            + _total_seconds(labels, "pipeline.adaln_cache_build")
        ),
        "dit_forward": _total_seconds(labels, "pipeline.dit_forward_step"),
        "video_vae_load_decode": (
            _total_seconds(labels, "load.video_vae_low_memory")
            + _total_seconds(labels, "pipeline.video_vae_decode")
        ),
        "audio_vae_load_decode": (
            _total_seconds(labels, "load.audio_vae_low_memory")
            + _total_seconds(labels, "pipeline.audio_vae_decode")
        ),
        "media_mux": _total_seconds(labels, "media.save_mp4_mux"),
    }
    stage_active_memory = {
        "text_conditioning": _max_observed_bytes(
            labels,
            ("load.text_encoder_low_memory", "pipeline.text_encoder_encode"),
            "max_active_bytes_observed",
        ),
        "dit_setup": _max_observed_bytes(
            labels,
            ("load.streaming_transformer_low_memory", "pipeline.adaln_cache_build"),
            "max_active_bytes_observed",
        ),
        "dit_forward": _max_observed_bytes(
            labels,
            ("pipeline.dit_forward_step",),
            "max_active_bytes_observed",
        ),
        "video_vae_load_decode": _max_observed_bytes(
            labels,
            ("load.video_vae_low_memory", "pipeline.video_vae_decode"),
            "max_active_bytes_observed",
        ),
        "audio_vae_load_decode": _max_observed_bytes(
            labels,
            ("load.audio_vae_low_memory", "pipeline.audio_vae_decode"),
            "max_active_bytes_observed",
        ),
        "media_mux": _max_observed_bytes(
            labels,
            ("media.save_mp4_mux",),
            "max_active_bytes_observed",
        ),
    }

    forward_wall = _total_seconds(labels, "pipeline.dit_forward_step") or _total_seconds(categories, "dit_forward_total")
    linear_layers = _total_seconds(categories, "linear_projection")
    attention = _total_seconds(categories, "attention_sdpa")
    if forward_wall:
        other_unattributed = forward_wall - linear_layers - attention
    else:
        other_unattributed = (
            _total_seconds(categories, "adaln_norm_residual")
            + _total_seconds(categories, "ffn_swiglu")
            + _total_seconds(categories, "final_heads")
            + _total_seconds(categories, "other_forward")
            + lazy_block_load
        )
    measured_other_subcomponents = {
        "adaln_norm_residual": _total_seconds(categories, "adaln_norm_residual"),
        "ffn_swiglu": _total_seconds(categories, "ffn_swiglu"),
        "final_heads": _total_seconds(categories, "final_heads"),
        "other_forward": _total_seconds(categories, "other_forward"),
        "lazy_streaming_block_load": lazy_block_load,
    }
    measured_other_total = sum(measured_other_subcomponents.values())

    bucket_total = linear_layers + attention + other_unattributed
    profile_non_overlapping = profile_summary.get("non_overlapping_share_denominator_seconds")
    total_wall = result.get("total_wall_time_seconds") or generation.get("elapsed_wall_seconds_python")
    profile_gap = None
    if isinstance(total_wall, (int, float)) and isinstance(profile_non_overlapping, (int, float)):
        profile_gap = float(total_wall) - float(profile_non_overlapping)

    media_validation_payload = result.get("media_validation", {}) or {}
    mp4_path = media_validation_payload.get("path") or generation.get("output")
    wav_path = str(Path(mp4_path).with_suffix(".wav")) if mp4_path else None
    wav_exists = Path(wav_path).exists() if wav_path else False

    metrics = generation.get("metrics", {}) or {}
    stdout_metrics = generation.get("stdout_metrics", {}) or {}
    memory_delta_payload = (generation.get("memory", {}) or {}).get("delta", {}) or {}
    return {
        "source": "derived_from_real_profiled_generation_artifact",
        "total_wall_time_seconds": total_wall,
        "load_init_overhead_seconds": load_init_total,
        "load_init_components_seconds": load_init_components,
        "pipeline_stage_timing_seconds": stage_timing,
        "pipeline_stage_max_active_memory_observed_bytes": stage_active_memory,
        "pipeline_stage_memory_semantics": (
            "Maximum MLX active allocation observed at synchronized profiler boundaries within each "
            "stage; this is not a per-stage peak. Use resource_deltas.mlx_peak_bytes for the process-wide peak."
        ),
        "generation_forward_wall_seconds": forward_wall,
        "block_cache_observed": stdout_metrics.get("block_cache"),
        "major_component_timing_seconds": {
            "linear_layers": linear_layers,
            "attention": attention,
            "other_unattributed": other_unattributed,
        },
        "other_unattributed_breakdown_seconds": {
            "measured_other_forward_subcomponents": measured_other_subcomponents,
            "measured_other_forward_subcomponents_total": measured_other_total,
            "residual_uninstrumented_or_wrapper_overhead": other_unattributed - measured_other_total,
        },
        "reconciliation": {
            "forward_wall_basis": "pipeline.dit_forward_step synchronized wrapper; nested category totals are not added to the wrapper total except via residual bucket assignment",
            "forward_bucket_total_seconds": bucket_total,
            "forward_bucket_minus_measured_wall_seconds": bucket_total - forward_wall if forward_wall else None,
            "end_to_end_non_overlapping_profile_seconds": profile_non_overlapping,
            "end_to_end_wall_minus_profile_seconds": profile_gap,
            "overlap_assumptions": [
                "profiled MLX regions use mx.synchronize before/after and evaluate MLX outputs before stopping timers",
                "dit_forward_total/denoise_step_total wrappers overlap nested component events and are excluded from non-overlapping shares",
                "other_unattributed is the residual needed to reconcile linear and attention buckets to the measured DiT forward wall",
                "end-to-end wall gap covers unprofiled Python/subprocess startup, import/setup, CLI bookkeeping, and measurement overhead",
            ],
        },
        "resource_deltas": {
            "mlx_peak_bytes": metrics.get("mlx_peak_bytes"),
            "max_rss_bytes": metrics.get("max_rss_bytes"),
            "pageouts_delta": memory_delta_payload.get("pageouts"),
            "swapouts_delta": memory_delta_payload.get("swapouts"),
            "pageouts_bytes_delta": memory_delta_payload.get("pageouts_bytes"),
            "swapouts_bytes_delta": memory_delta_payload.get("swapouts_bytes"),
            "process_swaps": metrics.get("process_swaps"),
        },
        "generated_media": {
            "mp4_path": mp4_path,
            "wav_path": wav_path,
            "mp4_valid": media_validation_payload.get("valid"),
            "wav_exists": wav_exists,
            "ffprobe_path": media_validation_payload.get("ffprobe_path"),
            "audio_activity_ok": (media_validation_payload.get("audio", {}) or {}).get("activity_ok"),
            "decoded_video_frames": (media_validation_payload.get("video", {}) or {}).get("decoded_frame_count"),
        },
    }


def build_generation_command(args: argparse.Namespace, run_dir: Path, resolution: str) -> tuple[list[str], Path, Path]:
    logs = run_dir / "command_outputs"
    media = run_dir / "media" / f"profiled_default_dense_{resolution}.mp4"
    profile_json = run_dir / "forward_profile.json"
    cmd = [
        "/usr/bin/time",
        "-l",
        str(ROOT / ".venv/bin/python"),
        "scripts/run_with_mlx_memory.py",
        "scripts/generate.py",
        args.prompt,
        "--checkpoint",
        args.checkpoint,
        "--transformer",
        args.transformer,
        "--text-encoder",
        args.text_encoder,
        "--profile",
        "balanced",
        "--low-memory",
        "--stream-blocks",
        "--resolution",
        resolution,
        "--duration",
        str(args.duration),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--dense-dequant-profile",
        "off",
        "--ffmpeg",
        args.ffmpeg,
        "--require-muxed-mp4",
        "--memory-pressure-guard",
        "--forward-profile-json",
        str(profile_json),
        "--output",
        str(media),
    ]
    if getattr(args, "video_vae_skip_decode_sync", False):
        cmd.append("--video-vae-skip-decode-sync")
    if getattr(args, "video_vae_disable_decode_tiling", False):
        cmd.append("--video-vae-disable-decode-tiling")
    decoder_quantization = getattr(args, "video_vae_decoder_quantization", "off")
    if decoder_quantization != "off":
        cmd.extend(["--video-vae-decoder-quantization", decoder_quantization])
    video_vae_precision = getattr(args, "video_vae_precision", "fp32")
    if video_vae_precision != "fp32":
        cmd.extend(["--video-vae-precision", video_vae_precision])
    if args.block_cache:
        cmd.extend(
            [
                "--block-cache",
                "--block-cache-threshold",
                str(args.block_cache_threshold),
                "--block-cache-start-percent",
                str(args.block_cache_start_percent),
                "--block-cache-end-percent",
                str(args.block_cache_end_percent),
                "--block-cache-depth",
                str(args.block_cache_depth),
                "--block-cache-max-consecutive",
                str(args.block_cache_max_consecutive),
            ]
        )
    else:
        cmd.append("--no-block-cache")
    return cmd, media, profile_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/MiniMax-H3/FL2VA")
    parser.add_argument("--transformer", default="models/MiniMax-H3-MLX-4bit")
    parser.add_argument("--text-encoder", default="models/MiniMax-H3/FL2VA/text_encoder-mlx-4bit")
    parser.add_argument("--ffmpeg", default=".venv/bin/static_ffmpeg")
    parser.add_argument("--ffprobe", default=".venv/bin/static_ffprobe")
    parser.add_argument("--prompt", default="A small orange cat walks across a desk")
    parser.add_argument("--resolution", default="64x64", help="bounded real rung to profile; use 320x192 only after healthy preflight")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--block-cache", action="store_true", help="enable the opt-in residual tail block cache for this profiled generation")
    parser.add_argument(
        "--video-vae-skip-decode-sync",
        action="store_true",
        help="enable the default-off VideoVAE decode candidate that skips internal mx.eval boundaries",
    )
    parser.add_argument(
        "--video-vae-disable-decode-tiling",
        action="store_true",
        help="enable the default-off VideoVAE decode candidate that uses one full spatial decode grid",
    )
    parser.add_argument(
        "--video-vae-decoder-quantization",
        choices=("off", "8bit", "4bit"),
        default="off",
        help="enable default-off VideoVAE decoder QuantizedLinear loading for profiled generation",
    )
    parser.add_argument(
        "--video-vae-precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="enable default-off VideoVAE lower-precision parameter loading/decode for profiled generation",
    )
    parser.add_argument("--block-cache-threshold", type=float, default=0.12)
    parser.add_argument("--block-cache-start-percent", type=float, default=0.10)
    parser.add_argument("--block-cache-end-percent", type=float, default=0.90)
    parser.add_argument("--block-cache-depth", type=float, default=0.75)
    parser.add_argument("--block-cache-max-consecutive", type=int, default=2)
    parser.add_argument("--preflight-idle-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--allow-unhealthy-preflight",
        action="store_true",
        help="run despite recorded memory/swap preflight issues; intended only for an explicitly supervised bounded profile",
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)
    if args.block_cache_threshold < 0:
        parser.error("--block-cache-threshold must be non-negative")
    if not 0.0 <= args.block_cache_start_percent <= args.block_cache_end_percent <= 1.0:
        parser.error("--block-cache-start-percent/end-percent must satisfy 0 <= start <= end <= 1")
    if not 0.0 <= args.block_cache_depth < 1.0:
        parser.error("--block-cache-depth must satisfy 0 <= depth < 1")
    if args.block_cache_max_consecutive < 0:
        parser.error("--block-cache-max-consecutive must be non-negative")

    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "experiments" / f"forward_pass_profile_{utc_stamp()}"
    logs = run_dir / "command_outputs"
    memory_dir = run_dir / "memory"
    (run_dir / "media").mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": iso_now(),
        "run_dir": str(run_dir),
        "objective": "full MiniMax-H3 MLX generation forward-pass profile plus activation quantization support probe",
        "environment": command_versions(),
        "paths": {
            "checkpoint": args.checkpoint,
            "transformer": args.transformer,
            "text_encoder": args.text_encoder,
            "ffmpeg": args.ffmpeg,
            "ffprobe": args.ffprobe,
        },
        "cli_options": {
            "profile": "balanced",
            "low_memory": True,
            "stream_blocks": True,
            "block_cache": bool(args.block_cache),
            "block_cache_threshold": args.block_cache_threshold,
            "block_cache_start_percent": args.block_cache_start_percent,
            "block_cache_end_percent": args.block_cache_end_percent,
            "block_cache_depth": args.block_cache_depth,
            "block_cache_max_consecutive": args.block_cache_max_consecutive,
            "dense_dequant_profile": "off",
            "memory_pressure_guard": True,
            "video_vae_skip_decode_sync": bool(args.video_vae_skip_decode_sync),
            "video_vae_disable_decode_tiling": bool(args.video_vae_disable_decode_tiling),
            "video_vae_decoder_quantization": args.video_vae_decoder_quantization,
            "video_vae_precision": args.video_vae_precision,
            "resolution": args.resolution,
            "duration_seconds": args.duration,
            "steps_sigma_points": args.steps,
            "seed": args.seed,
            "allow_unhealthy_preflight": args.allow_unhealthy_preflight,
        },
    }

    result["preflight"] = preflight(memory_dir, idle_seconds=args.preflight_idle_seconds)
    needs_healthy_preflight = resolution_requires_healthy_preflight(args.resolution)
    result["preflight_override_used"] = bool(
        needs_healthy_preflight and not result["preflight"].get("healthy") and args.allow_unhealthy_preflight
    )
    if needs_healthy_preflight and not result["preflight"].get("healthy") and not args.allow_unhealthy_preflight:
        result["generation_attempted"] = False
        result["blocker_fingerprint"] = "blocker:forward-profile-resolution-preflight-unhealthy-v1"
        result["quantized_activation_support"] = quantized_activation_probe()
        (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"run_dir": str(run_dir), "status": "blocked_preflight", "issues": result["preflight"].get("issues")}, indent=2))
        return 2

    before = memory_sample("pre_generation", memory_dir)
    cmd, media_path, profile_json = build_generation_command(args, run_dir, args.resolution)
    generation = run_capture(
        cmd,
        stdout=logs / "profiled_generation.stdout",
        stderr=logs / "profiled_generation.stderr",
        timeout=args.timeout_seconds,
    )
    after = memory_sample("post_generation", memory_dir)
    generation["metrics"] = parse_time_l(Path(generation["stderr_path"]).read_text())
    generation["stdout_metrics"] = parse_generation_stdout(Path(generation["stdout_path"]).read_text())
    generation["memory"] = {"pre": before, "post": after, "delta": memory_delta(before, after)}
    generation["output"] = str(media_path)
    generation["forward_profile_json"] = str(profile_json)
    generation["forward_profile_loaded"] = profile_json.exists()
    if profile_json.exists():
        try:
            profile_payload = json.loads(profile_json.read_text())
            generation["forward_profile_summary"] = profile_payload.get("summary")
        except Exception as exc:
            generation["forward_profile_parse_error"] = f"{type(exc).__name__}: {exc}"
    result["generation_attempted"] = True
    result["generation"] = generation
    result["total_wall_time_seconds"] = generation["elapsed_wall_seconds_python"]

    result["media_validation"] = media_validation(
        media_path,
        ffmpeg=Path(args.ffmpeg),
        ffprobe=Path(args.ffprobe),
        logs=logs,
    )
    result["quantized_activation_support"] = quantized_activation_probe()
    result["acceptance_summary"] = build_acceptance_summary(result)
    result["resource_usage_self"] = {"ru_maxrss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    result["status"] = "ok" if generation["returncode"] == 0 and result["media_validation"].get("valid") else "failed"

    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "status": result["status"], "result": str(run_dir / "result.json")}, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
