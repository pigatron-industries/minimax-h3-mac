"""Parser-only regression tests for scripts/generate.py.

These tests never load MiniMax-H3 weights or run generation. They verify the
release-facing CLI contract: no machine-local checkpoint default, profile and
resolution parsing, legacy width/height compatibility, and --steps semantics.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_generate():
    path = ROOT / "scripts" / "generate.py"
    spec = importlib.util.spec_from_file_location("generate_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_case(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"{name} failed{(': ' + detail) if detail else ''}")
    print(f"ok  {name}{(' — ' + detail) if detail else ''}")


def assert_parse_error(module, argv: list[str], *, env: dict[str, str] | None = None) -> None:
    try:
        module.parse_args(argv, env={} if env is None else env)
    except SystemExit as exc:
        assert_case(f"parse error for {' '.join(argv)}", exc.code == 2, f"exit={exc.code}")
        return
    raise AssertionError(f"expected parse error for {argv!r}")


def test_missing_checkpoint_has_no_local_absolute_default() -> None:
    module = load_generate()
    source = (ROOT / "scripts" / "generate.py").read_text()
    parser = module.build_parser()
    checkpoint_actions = [action for action in parser._actions if "--checkpoint" in action.option_strings]
    assert_case("checkpoint option exists", len(checkpoint_actions) == 1)
    assert_case("checkpoint default is None", checkpoint_actions[0].default is None)
    old_machine_local_default = "/" + "Volumes/models"
    assert_case("old /Volumes model default absent", old_machine_local_default not in source)
    assert_parse_error(module, ["a safe prompt"], env={})
    args = module.parse_args(["a safe prompt"], env={module.CHECKPOINT_ENV_VAR: "models/MiniMax-H3/FL2VA"})
    assert_case("checkpoint may be sourced from documented env", args.checkpoint == "models/MiniMax-H3/FL2VA")


def test_profile_presets_parse_and_explicit_flags_win() -> None:
    module = load_generate()
    contracts = module.profile_contracts()
    assert_case("three release profiles exposed", set(contracts) == {"balanced", "quality", "speed"})
    speed = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--profile", "speed"],
        env={},
    )
    assert_case("speed profile parsed", speed.profile == "speed")
    assert_case("speed profile applies preset steps", speed.steps == contracts["speed"]["steps_sigma_points"])
    assert_case("speed profile enables block cache", speed.block_cache is True)
    assert_case("text conditioning cache remains opt-in", speed.cache_text_conditioning is False)
    assert_case("memory pressure guard remains opt-in", speed.memory_pressure_guard is False)
    assert_case("VideoVAE decode sync-skip remains opt-in", speed.video_vae_skip_decode_sync is False)
    assert_case("VideoVAE decode tiling remains enabled by default", speed.video_vae_disable_decode_tiling is False)
    assert_case("VideoVAE decoder quantization remains opt-in", speed.video_vae_decoder_quantization == "off")
    assert_case("VideoVAE lower precision remains opt-in", speed.video_vae_precision == "fp32")
    assert_case("stream block group size keeps default one-block residency", speed.stream_block_group_size == 1)
    assert_case("block cache middle-window start default parses", speed.block_cache_start_percent == 0.10)
    assert_case("block cache middle-window end default parses", speed.block_cache_end_percent == 0.90)
    assert_case("dense-dequant generation profile remains opt-in", speed.dense_dequant_profile == "off")
    parser = module.build_parser()
    block_loader_actions = [action for action in parser._actions if "--block-load-mode" in action.option_strings]
    assert_case("rejected selective block loader is absent from generation CLI", block_loader_actions == [])
    assert_parse_error(
        module,
        ["a safe prompt", "--checkpoint", "models/upstream", "--block-load-mode", "selective_safetensors"],
        env={},
    )
    text_cache = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--cache-text-conditioning"],
        env={},
    )
    assert_case("text conditioning cache can be requested", text_cache.cache_text_conditioning is True)
    memory_guard = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--memory-pressure-guard"],
        env={},
    )
    assert_case("memory pressure guard can be requested", memory_guard.memory_pressure_guard is True)
    decode_sync = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-skip-decode-sync"],
        env={},
    )
    assert_case("VideoVAE decode sync-skip can be requested", decode_sync.video_vae_skip_decode_sync is True)
    decode_full_grid = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-disable-decode-tiling"],
        env={},
    )
    assert_case("VideoVAE full-grid decode can be requested", decode_full_grid.video_vae_disable_decode_tiling is True)
    decoder_quantized = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-decoder-quantization", "8bit"],
        env={},
    )
    assert_case("VideoVAE decoder 8-bit quantization can be requested", decoder_quantized.video_vae_decoder_quantization == "8bit")
    assert_parse_error(
        module,
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-decoder-quantization", "int3"],
        env={},
    )
    vae_bf16 = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-precision", "bf16"],
        env={},
    )
    assert_case("VideoVAE BF16 precision can be requested", vae_bf16.video_vae_precision == "bf16")
    vae_fp16 = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-precision", "fp16"],
        env={},
    )
    assert_case("VideoVAE FP16 precision can be requested", vae_fp16.video_vae_precision == "fp16")
    assert_parse_error(
        module,
        ["a safe prompt", "--checkpoint", "models/upstream", "--video-vae-precision", "fp64"],
        env={},
    )
    grouped_stream = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--stream-block-group-size", "2"],
        env={},
    )
    assert_case("stream block group-size opt-in parses", grouped_stream.stream_block_group_size == 2)
    assert_parse_error(
        module,
        ["a safe prompt", "--checkpoint", "models/upstream", "--stream-block-group-size", "0"],
        env={},
    )
    cache_window = module.parse_args(
        [
            "a safe prompt",
            "--checkpoint",
            "models/upstream",
            "--block-cache",
            "--block-cache-start-percent",
            "0.25",
            "--block-cache-end-percent",
            "0.75",
        ],
        env={},
    )
    assert_case("block cache start/end window can be requested", cache_window.block_cache_start_percent == 0.25 and cache_window.block_cache_end_percent == 0.75)
    assert_parse_error(
        module,
        [
            "a safe prompt",
            "--checkpoint",
            "models/upstream",
            "--block-cache-start-percent",
            "0.8",
            "--block-cache-end-percent",
            "0.2",
        ],
        env={},
    )
    dense_profile = module.parse_args(
        [
            "a safe prompt",
            "--checkpoint",
            "models/upstream",
            "--dense-dequant-profile",
            "ffn-fc2-tiled",
            "--ffn-fc2-tile-size",
            "1024",
        ],
        env={},
    )
    assert_case("fc2-only dense-dequant opt-in can be requested", dense_profile.dense_dequant_profile == "ffn-fc2-tiled")
    assert_case("dense-dequant opt-in fc2 tile parses", dense_profile.ffn_fc2_tile_size == 1024)
    forward_profile = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--forward-profile-json", "experiments/run/forward_profile.json"],
        env={},
    )
    assert_case("forward profiling remains disabled unless a JSON path is requested", forward_profile.forward_profile_json.endswith("forward_profile.json"))
    default_profile = module.parse_args(["a safe prompt", "--checkpoint", "models/upstream"], env={})
    assert_case("forward profiling default is off", default_profile.forward_profile_json is None)
    overridden = module.parse_args(
        [
            "a safe prompt",
            "--checkpoint",
            "models/upstream",
            "--profile",
            "speed",
            "--steps",
            "9",
            "--no-block-cache",
            "--no-low-memory",
        ],
        env={},
    )
    assert_case("explicit --steps overrides profile", overridden.steps == 9)
    assert_case("explicit --no-block-cache overrides profile", overridden.block_cache is False)
    assert_case("explicit --no-low-memory overrides profile", overridden.low_memory is False)


def test_resolution_parses_validates_and_conflicts_with_legacy_surface() -> None:
    module = load_generate()
    args = module.parse_args(
        ["a safe prompt", "--checkpoint", "models/upstream", "--resolution", "320x192"],
        env={},
    )
    assert_case("resolution sets width", args.width == 320)
    assert_case("resolution sets height", args.height == 192)
    assert_parse_error(
        module,
        ["a safe prompt", "--checkpoint", "models/upstream", "--resolution", "320x192", "--width", "320"],
        env={},
    )
    assert_parse_error(
        module,
        ["a safe prompt", "--checkpoint", "models/upstream", "--resolution", "321x192"],
        env={},
    )


def test_generation_cli_exposes_no_macsol_deployable_surface() -> None:
    module = load_generate()
    parser = module.build_parser()
    option_strings = [option for action in parser._actions for option in action.option_strings]
    forbidden_tokens = (
        "macsol",
        "sol-attn",
        "sol_attn",
        "budget-topk",
        "budget_topk",
        "topk",
        "threshold-h3-structure",
        "threshold_h3_structure",
        "spatial-tube",
        "spatial_tube",
    )
    leaked = [opt for opt in option_strings if any(token in opt.lower() for token in forbidden_tokens)]
    assert_case("no MacSol/Sol-Attn deployable generation flags", leaked == [], f"leaked={leaked}")


def test_legacy_width_height_and_steps_semantics_are_preserved() -> None:
    module = load_generate()
    args = module.parse_args(
        [
            "a safe prompt",
            "--checkpoint",
            "models/upstream",
            "--width",
            "960",
            "--height",
            "544",
            "--steps",
            "5",
        ],
        env={},
    )
    assert_case("legacy width parsed", args.width == 960)
    assert_case("legacy height parsed", args.height == 544)
    assert_case("steps arg remains sigma grid points", args.steps == 5)
    contract = module.generation_schedule_contract(args.steps)
    assert_case("5 sigma points maps to 4 NFE", contract["denoiser_evaluations"] == 4)
    assert_parse_error(module, ["a safe prompt", "--checkpoint", "models/upstream", "--steps", "1"], env={})


def main() -> int:
    test_missing_checkpoint_has_no_local_absolute_default()
    test_profile_presets_parse_and_explicit_flags_win()
    test_resolution_parses_validates_and_conflicts_with_legacy_surface()
    test_generation_cli_exposes_no_macsol_deployable_surface()
    test_legacy_width_height_and_steps_semantics_are_preserved()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
