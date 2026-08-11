from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import minimax_h3_mlx.forward_profile as module


ROOT = Path(__file__).resolve().parents[1]


def _load_runner_module():
    path = ROOT / "scripts" / "run_forward_profile_experiment.py"
    spec = importlib.util.spec_from_file_location("run_forward_profile_experiment_under_test", path)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


def test_profiler_records_stage_memory(monkeypatch):
    snapshots = iter(
        [
            {"active_bytes": 10, "cache_bytes": 2, "peak_bytes": 10},
            {"active_bytes": 30, "cache_bytes": 4, "peak_bytes": 32},
        ]
    )
    monkeypatch.setattr(module, "_mlx_memory_snapshot", lambda: next(snapshots))

    profiler = module.ForwardPassProfiler(synchronize=False)
    assert profiler.call("load.model", "load_overhead", lambda: "loaded") == "loaded"

    event = profiler.events[0]
    assert event["memory_before"]["active_bytes"] == 10
    assert event["memory_after"]["active_bytes"] == 30
    assert event["active_memory_delta_bytes"] == 20

    row = profiler.summary()["label_totals"]["load.model"]
    assert row["max_active_bytes_observed"] == 30
    assert row["max_cache_bytes_observed"] == 4
    assert row["max_peak_bytes_observed"] == 32


def test_profiler_block_records_stage_memory(monkeypatch):
    snapshots = iter(
        [
            {"active_bytes": 40, "cache_bytes": 8, "peak_bytes": 40},
            {"active_bytes": 25, "cache_bytes": 3, "peak_bytes": 41},
        ]
    )
    monkeypatch.setattr(module, "_mlx_memory_snapshot", lambda: next(snapshots))

    profiler = module.ForwardPassProfiler(synchronize=False)
    with profiler.block("media.mux", "media_mux"):
        pass

    event = profiler.events[0]
    assert event["active_memory_delta_bytes"] == -15
    assert profiler.summary()["category_totals"]["media_mux"]["max_peak_bytes_observed"] == 41


def test_generation_stdout_parser_extracts_block_cache_stats():
    runner = _load_runner_module()

    parsed = runner.parse_generation_stdout(
        "12.3s per step, 0.8 min total\n"
        "block cache: full=2 cached=2 skipped=74 blocks (37.0% of block executions)\n"
    )

    assert parsed["pipeline_seconds_per_step"] == 12.3
    assert parsed["block_cache"] == {
        "full_steps": 2,
        "cache_steps": 2,
        "skipped_blocks": 74,
        "saved_fraction_percent": 37.0,
    }


def test_acceptance_summary_reports_stage_boundaries():
    runner = _load_runner_module()
    labels = {
        "load.text_encoder_low_memory": {
            "total_seconds": 1.0,
            "max_active_bytes_observed": 10,
        },
        "pipeline.text_encoder_encode": {
            "total_seconds": 2.0,
            "max_active_bytes_observed": 30,
        },
        "load.streaming_transformer_low_memory": {
            "total_seconds": 3.0,
            "max_active_bytes_observed": 20,
        },
        "pipeline.adaln_cache_build": {
            "total_seconds": 4.0,
            "max_active_bytes_observed": 40,
        },
        "pipeline.dit_forward_step": {
            "total_seconds": 5.0,
            "max_active_bytes_observed": 50,
        },
    }
    result = {
        "generation": {
            "forward_profile_summary": {
                "label_totals": labels,
                "category_totals": {},
            },
            "metrics": {},
            "stdout_metrics": {"block_cache": {"full_steps": 2, "cache_steps": 1, "skipped_blocks": 37}},
            "memory": {"delta": {}},
        },
        "media_validation": {},
    }

    summary = runner.build_acceptance_summary(result)

    assert summary["pipeline_stage_timing_seconds"]["text_conditioning"] == 3.0
    assert summary["pipeline_stage_timing_seconds"]["dit_setup"] == 7.0
    assert summary["block_cache_observed"] == {"full_steps": 2, "cache_steps": 1, "skipped_blocks": 37}
    assert summary["pipeline_stage_max_active_memory_observed_bytes"]["text_conditioning"] == 30
    assert summary["pipeline_stage_max_active_memory_observed_bytes"]["dit_setup"] == 40
    assert summary["pipeline_stage_max_active_memory_observed_bytes"]["dit_forward"] == 50


def test_process_preflight_ignores_repository_git_transport():
    runner = _load_runner_module()

    assert not runner.is_h3_mlx_process(
        "33016 33012 6144 git-remote-https origin https://github.com/Argus-AiTeam/minimax-h3-mac.git"
    )
    assert runner.is_h3_mlx_process(
        "41000 1 123456 .venv/bin/python scripts/run_with_mlx_memory.py scripts/generate.py prompt"
    )


def test_generation_command_exposes_video_vae_decode_sync_opt_in(tmp_path):
    runner = _load_runner_module()
    args = SimpleNamespace(
        prompt="a safe prompt",
        checkpoint="models/upstream",
        transformer="models/dit",
        text_encoder="models/text",
        duration=5.0,
        steps=5,
        seed=7,
        ffmpeg=".venv/bin/static_ffmpeg",
        block_cache=False,
        video_vae_skip_decode_sync=True,
        video_vae_disable_decode_tiling=True,
        video_vae_decoder_quantization="off",
        video_vae_precision="fp16",
    )

    cmd, _, _ = runner.build_generation_command(args, tmp_path, "320x192")

    assert "--video-vae-skip-decode-sync" in cmd
    assert "--video-vae-disable-decode-tiling" in cmd
    assert cmd[cmd.index("--video-vae-precision") + 1] == "fp16"
    assert "--no-block-cache" in cmd


def test_resolution_preflight_gate_uses_320x192_area():
    runner = _load_runner_module()

    assert not runner.resolution_requires_healthy_preflight("256x160")
    assert runner.resolution_requires_healthy_preflight("320x192")
    assert runner.resolution_requires_healthy_preflight("1344x768")
