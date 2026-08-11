"""Video VAE memory boundaries must change execution timing, not values."""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig


def config() -> VideoVAEConfig:
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


def main() -> None:
    mx.random.seed(0)
    model = VideoVAE(config())
    mx.eval(model.parameters())
    model.tile_sample_min_height = model.tile_sample_min_width = 8
    model.tile_sample_min_overlap_height = model.tile_sample_min_overlap_width = 4
    latents = mx.random.normal((1, 2, 8, 8, 4))
    mx.eval(latents)

    eager_boundaries = model._decode_clip(latents)
    mx.eval(eager_boundaries)
    assert model.decode_internal_eval_boundaries is True
    assert model.decoder.internal_eval_boundaries is True

    model.set_decode_internal_eval_boundaries(False)
    lazy_graph = model._decode_clip(latents)
    mx.eval(lazy_graph)

    delta = float(mx.max(mx.abs(eager_boundaries - lazy_graph)).item())
    assert delta == 0.0, delta
    assert eager_boundaries.shape == (1, 2, 16, 16, 3)
    assert model.decode_internal_eval_boundaries is False
    assert model.decoder.internal_eval_boundaries is False
    print(f"video VAE decode sync opt-in exact: shape={eager_boundaries.shape}, delta={delta}")


if __name__ == "__main__":
    main()
