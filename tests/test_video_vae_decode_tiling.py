from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig


def tiny_config() -> VideoVAEConfig:
    return VideoVAEConfig(
        latent_channels=4,
        block_out_channels=(8,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(1,),
        norm_num_groups=1,
        decoder_num_layers=2,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=12,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=2,
        decoder_rope_dim_ratio=0.5,
        clip_length=5,
        token_drop=1,
    )


def test_decode_spatial_tiling_switch_is_default_on_and_decode_only(monkeypatch) -> None:
    mx.random.seed(0)
    model = VideoVAE(tiny_config())
    mx.eval(model.parameters())
    model.tile_sample_min_height = model.tile_sample_min_width = 8
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 4

    latents = mx.random.normal((1, 2, 8, 12, 4))
    mx.eval(latents)

    original_split_tiles = model._split_tiles
    split_calls: list[tuple[int, int, int]] = []

    def spy_split_tiles(length: int, tile_size: int, min_overlap: int):
        split_calls.append((length, tile_size, min_overlap))
        return original_split_tiles(length, tile_size, min_overlap)

    monkeypatch.setattr(model, "_split_tiles", spy_split_tiles)
    default_tiled = model._decode_clip(latents)
    mx.eval(default_tiled)
    assert model.decode_spatial_tiling is True
    assert split_calls, "default decode should use the historical spatial tiling path"

    split_calls.clear()
    model.set_decode_spatial_tiling(True)
    explicit_default_tiled = model._decode_clip(latents)
    mx.eval(explicit_default_tiled)
    assert split_calls, "explicitly enabling decode tiling should still use the tiled path"
    assert float(mx.max(mx.abs(default_tiled - explicit_default_tiled)).item()) == 0.0

    def fail_if_split_tiles_is_used(*args, **kwargs):  # pragma: no cover - only used on failure
        raise AssertionError("full-grid decode must not call _split_tiles")

    model.set_decode_spatial_tiling(False)
    monkeypatch.setattr(model, "_split_tiles", fail_if_split_tiles_is_used)
    full_grid = model._decode_clip(latents)
    mx.eval(full_grid)
    assert full_grid.shape == default_tiled.shape

    split_calls.clear()
    monkeypatch.setattr(model, "_split_tiles", spy_split_tiles)
    pixels = mx.random.normal((1, 2, 16, 24, 3))
    encoded = model._encode_clip(pixels)
    mx.eval(encoded)
    assert split_calls, "disabling decode tiling must not disable encoder tiling"
    assert model.decode_spatial_tiling is False
