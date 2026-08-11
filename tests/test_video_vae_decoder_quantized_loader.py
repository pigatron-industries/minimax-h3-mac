from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from minimax_h3_mlx.load import load_video_vae, video_vae_parameter_dtype_summary
from minimax_h3_mlx.video_vae import VideoVAE, VideoVAEConfig


def tiny_config() -> VideoVAEConfig:
    return VideoVAEConfig(
        latent_channels=4,
        block_out_channels=(32,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(1,),
        norm_num_groups=1,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=32,
        decoder_num_register_tokens=1,
        decoder_ffn_mult=4,
        decoder_rope_dim_ratio=0.75,
        clip_length=5,
        token_drop=1,
    )


def write_tiny_video_vae_checkpoint(root: Path) -> None:
    cfg = tiny_config()
    source = root / "source"
    source.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "vae_clip_length": cfg.clip_length,
                "vae_token_drop": cfg.token_drop,
                "latents_mean": [0.0] * cfg.latent_channels,
                "latents_std": [1.0] * cfg.latent_channels,
            }
        )
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "in_channels": cfg.in_channels,
                "out_ch": cfg.out_channels,
                "z_channels": cfg.latent_channels,
                "ch": cfg.block_out_channels[0],
                "ch_mult": [1],
                "num_res_blocks": cfg.layers_per_block,
                "space_down": list(cfg.spatial_downsample_factors),
                "time_down": list(cfg.temporal_downsample_factors),
                "vit_decoder_kwargs": {
                    "num_layers": cfg.decoder_num_layers,
                    "heads": cfg.decoder_num_attention_heads,
                    "dim_head": cfg.decoder_attention_head_dim,
                    "rope_theta": cfg.decoder_rope_theta,
                    "rope_dim_ratio": cfg.decoder_rope_dim_ratio,
                },
            }
        )
    )

    mx.random.seed(123)
    model = VideoVAE(cfg)
    mx.eval(model.parameters())
    raw = {}
    for key, tensor in tree_flatten(model.parameters()):
        # load_video_vae expects torch-style 3D-conv source weights and transposes them to MLX.
        raw[key] = tensor.transpose(0, 4, 1, 2, 3) if tensor.ndim == 5 else tensor
    mx.save_safetensors(str(source / "model.safetensors"), raw)


def test_video_vae_decoder_quantized_loader_is_default_off_and_runs(tmp_path: Path) -> None:
    write_tiny_video_vae_checkpoint(tmp_path)

    unquantized = load_video_vae(tmp_path)
    assert isinstance(unquantized.decoder.transformer_blocks[0].attn.to_qkv, nn.Linear)
    assert getattr(unquantized, "video_vae_decoder_quantization") == {"enabled": False}
    assert getattr(unquantized, "video_vae_precision")["mode"] == "fp32"
    assert unquantized.decode_precision == "fp32"
    unquantized_params = dict(tree_flatten(unquantized.parameters()))
    assert unquantized_params["decoder.transformer_blocks.0.attn.to_qkv.weight"].dtype == mx.float32
    assert unquantized.video_vae_parameter_summary == video_vae_parameter_dtype_summary(unquantized)

    quantized = load_video_vae(tmp_path, decoder_quantization="8bit")
    assert isinstance(quantized.decoder.transformer_blocks[0].attn.to_qkv, nn.QuantizedLinear)
    assert isinstance(quantized.decoder.transformer_blocks[0].attn.to_out, nn.QuantizedLinear)
    assert isinstance(quantized.decoder.transformer_blocks[0].ff.w1, nn.QuantizedLinear)
    assert isinstance(quantized.decoder.transformer_blocks[0].ff.w2, nn.QuantizedLinear)
    assert isinstance(quantized.decoder.proj_out, nn.QuantizedLinear)
    assert isinstance(quantized.decoder.x_embedder, nn.Linear)
    summary = quantized.video_vae_decoder_quantization
    assert summary["enabled"] is True
    assert summary["bits"] == 8
    assert summary["group_size"] == 64
    assert summary["quantized_linear_count"] == 5

    params = dict(tree_flatten(quantized.parameters()))
    prefix = "decoder.transformer_blocks.0.attn.to_qkv"
    assert params[f"{prefix}.weight"].dtype == mx.uint32
    assert f"{prefix}.scales" in params
    assert f"{prefix}.biases" in params
    assert f"{prefix}.bias" in params

    latents = mx.array(np.random.default_rng(0).standard_normal((1, 4, 9, 4, 4)).astype(np.float32))
    decoded = quantized.decode(latents)
    mx.eval(decoded)
    assert decoded.shape == (1, 3, 9, 8, 8)


def test_video_vae_lower_precision_casts_floating_parameters_and_decodes(tmp_path: Path) -> None:
    write_tiny_video_vae_checkpoint(tmp_path)

    expectations = {
        "bf16": (mx.bfloat16, "bfloat16"),
        "fp16": (mx.float16, "float16"),
    }
    for precision, (dtype, input_name) in expectations.items():
        vae = load_video_vae(tmp_path, precision=precision)
        params = dict(tree_flatten(vae.parameters()))
        floating_dtypes = {tensor.dtype for tensor in params.values() if tensor.dtype in (mx.float16, mx.float32, mx.bfloat16)}
        assert floating_dtypes == {dtype}
        assert vae.decode_precision == precision
        assert vae.video_vae_precision["mode"] == precision
        assert vae.video_vae_precision["parameter_cast"] is True
        assert vae.video_vae_precision["decode_input_dtype"] == input_name
        assert vae.video_vae_parameter_summary["by_dtype"] == {str(dtype): vae.video_vae_parameter_summary["by_dtype"][str(dtype)]}

        latents = mx.array(np.random.default_rng(0).standard_normal((1, 4, 9, 4, 4)).astype(np.float32))
        decoded = vae.decode(latents)
        mx.eval(decoded)
        assert decoded.shape == (1, 3, 9, 8, 8)
        assert decoded.dtype == mx.float32
        assert np.array(decoded, dtype=np.float32).shape == (1, 3, 9, 8, 8)


def test_video_vae_decoder_quantized_loader_rejects_unknown_mode(tmp_path: Path) -> None:
    write_tiny_video_vae_checkpoint(tmp_path)
    try:
        load_video_vae(tmp_path, decoder_quantization="3bit")
    except ValueError as exc:
        assert "unknown VideoVAE decoder quantization" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid decoder quantization to fail")


def test_video_vae_precision_rejects_unknown_mode(tmp_path: Path) -> None:
    write_tiny_video_vae_checkpoint(tmp_path)
    try:
        load_video_vae(tmp_path, precision="fp64")
    except ValueError as exc:
        assert "unknown VideoVAE precision" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected invalid VideoVAE precision to fail")
