"""Checkpoint loading for the MiniMax-H3 MLX port.

The MLX module tree reproduces the original checkpoint names exactly, so loading is a 1:1 key
match — the only tensor the checkpoint carries that the port does not hold is ``rope.inv_freq``,
which is recomputed bit-identically from the config.

MiniMax-H3 ships a **mixed-precision** transformer: the two input patch projections, the timestep
MLP and the two output heads are float32 while everything else (including the AdaLN projections)
is bfloat16. That split is preserved on load — it is not incidental. The timestep MLP feeds every
block's modulation, so rounding it biases all 50 blocks identically at every sampling step and the
error accumulates coherently along the denoising trajectory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .dit import MiniMaxH3DiT

VIDEO_VAE_DECODER_QUANTIZATION_CHOICES = ("off", "8bit", "4bit")
VideoVAEDecoderQuantization = Literal["off", "8bit", "4bit"]
VIDEO_VAE_PRECISION_CHOICES = ("fp32", "bf16", "fp16")
VideoVAEPrecision = Literal["fp32", "bf16", "fp16"]
VIDEO_VAE_DECODER_QUANT_GROUP_SIZE = 64
VIDEO_VAE_DECODER_QUANT_MODE = "affine"

# Substring matches, mirroring the reference's `_keep_in_fp32_modules`.
FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# Carried by the checkpoint but recomputed by the port.
SKIP_KEYS = ("rope.inv_freq",)


def is_fp32_key(key: str) -> bool:
    return key.startswith(FP32_PREFIXES)


def shard_paths(model_dir: str | Path) -> list[Path]:
    """Resolve the safetensors shards of a transformer directory, in index order."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        names = sorted(set(weight_map.values()))
        return [model_dir / name for name in names]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}.")
    return shards


def load_dit(
    model_dir: str | Path,
    dtype: mx.Dtype | None = None,
    strict: bool = True,
    verbose: bool = False,
) -> MiniMaxH3DiT:
    """Load the 33B DiT from a released ``FL2VA/transformer`` (or ``Ref2VA/transformer``) directory.

    Args:
        model_dir: the transformer directory holding ``config.json`` and the shards.
        dtype: cast every tensor to this dtype. ``None`` (default) preserves the checkpoint's
            mixed float32/bfloat16 split, which is what the reference runs.
        strict: raise if the checkpoint and the module tree disagree on any key.
        verbose: print per-shard progress.

    Returns:
        A parameter-loaded :class:`MiniMaxH3DiT`.
    """
    model_dir = Path(model_dir)
    config = DiTConfig.from_json(model_dir / "config.json")
    model = MiniMaxH3DiT(config)

    # A quantized build carries `quant_config.json`. Quantized layers hold packed weights plus
    # scales and biases, so the module tree has to be quantized *before* loading or the keys will
    # not line up — the same recipe is replayed from the file rather than guessed.
    quant_path = model_dir / "quant_config.json"
    if quant_path.exists():
        from .quantize import QuantConfig, apply_quantization_structure

        with open(quant_path) as fh:
            recipe = json.load(fh)
        apply_quantization_structure(
            model,
            QuantConfig(
                bits=recipe["bits"],
                group_size=recipe["group_size"],
                quantize_adaln=recipe.get("quantize_adaln", False),
                adaln_bits=recipe.get("adaln_bits") or 8,
            ),
        )
        if verbose:
            print(f"  quantized structure: {recipe['bits']}-bit, group {recipe['group_size']}")

    expected = {key for key, _ in tree_flatten(model.parameters())}
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for shard in shard_paths(model_dir):
        started = time.perf_counter()
        loaded = mx.load(str(shard))
        for key, tensor in loaded.items():
            if key in SKIP_KEYS:
                continue
            if key not in expected:
                unexpected.append(key)
                continue
            if dtype is not None:
                # Bulk conversion of the whole 33B stack. Done on the CPU stream and materialized
                # per tensor: casting ~130 GB through Metal is enough submissions to trip the
                # command-buffer limits when anything else is using the device, and this path is
                # I/O-dominated anyway.
                with mx.stream(mx.cpu):
                    tensor = tensor.astype(dtype)
                    mx.eval(tensor)
            elif is_fp32_key(key) and tensor.dtype != mx.float32:
                # Only twelve small tensors; not worth a stream switch.
                tensor = tensor.astype(mx.float32)
            weights[key] = tensor
        if verbose:
            gb = sum(t.nbytes for t in loaded.values()) / 1e9
            print(f"  {shard.name}: {len(loaded)} tensors, {gb:.2f} GB, "
                  f"{time.perf_counter() - started:.1f}s")

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Checkpoint/module mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )

    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def read_video_vae_config(model_dir: str | Path):
    """Read Video VAE metadata without allocating model weights."""
    from .video_vae import VideoVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as handle:
        wrapper = json.load(handle)
    with open(model_dir / "source" / "config.json") as handle:
        source = json.load(handle)
    channels = source["ch"]
    return VideoVAEConfig(
        in_channels=source["in_channels"],
        out_channels=source["out_ch"],
        latent_channels=source["z_channels"],
        block_out_channels=tuple(channels * multiplier for multiplier in source["ch_mult"]),
        layers_per_block=source["num_res_blocks"],
        spatial_downsample_factors=tuple(source["space_down"]),
        temporal_downsample_factors=tuple(source["time_down"]),
        decoder_num_layers=source["vit_decoder_kwargs"]["num_layers"],
        decoder_num_attention_heads=source["vit_decoder_kwargs"]["heads"],
        decoder_attention_head_dim=source["vit_decoder_kwargs"]["dim_head"],
        decoder_rope_theta=source["vit_decoder_kwargs"]["rope_theta"],
        decoder_rope_dim_ratio=source["vit_decoder_kwargs"]["rope_dim_ratio"],
        clip_length=wrapper.get("vae_clip_length", 17),
        token_drop=wrapper.get("vae_token_drop", 3),
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def _video_vae_decoder_quant_bits(decoder_quantization: str | None) -> int | None:
    selected = "off" if decoder_quantization is None else str(decoder_quantization).strip().lower()
    aliases = {
        "": "off",
        "none": "off",
        "false": "off",
        "0": "off",
        "8": "8bit",
        "int8": "8bit",
        "q8": "8bit",
        "8-bit": "8bit",
        "4": "4bit",
        "int4": "4bit",
        "q4": "4bit",
        "4-bit": "4bit",
    }
    selected = aliases.get(selected, selected)
    if selected == "off":
        return None
    if selected == "8bit":
        return 8
    if selected == "4bit":
        return 4
    allowed = ", ".join(VIDEO_VAE_DECODER_QUANTIZATION_CHOICES)
    raise ValueError(f"unknown VideoVAE decoder quantization {decoder_quantization!r}; expected one of: {allowed}")


def _normalize_video_vae_precision(precision: str | None) -> str:
    selected = "fp32" if precision is None else str(precision).strip().lower()
    aliases = {
        "": "fp32",
        "default": "fp32",
        "source": "fp32",
        "float32": "fp32",
        "f32": "fp32",
        "bfloat16": "bf16",
        "bf-16": "bf16",
        "bfloat-16": "bf16",
        "float16": "fp16",
        "f16": "fp16",
        "half": "fp16",
        "float-16": "fp16",
    }
    selected = aliases.get(selected, selected)
    if selected in VIDEO_VAE_PRECISION_CHOICES:
        return selected
    allowed = ", ".join(VIDEO_VAE_PRECISION_CHOICES)
    raise ValueError(f"unknown VideoVAE precision {precision!r}; expected one of: {allowed}")


def _is_float_tensor(tensor: mx.array) -> bool:
    return tensor.dtype in (mx.float16, mx.float32, mx.bfloat16)


def _cast_video_vae_tensor(tensor: mx.array, precision: str) -> mx.array:
    target_dtype = {"bf16": mx.bfloat16, "fp16": mx.float16}.get(precision)
    if target_dtype is None or not _is_float_tensor(tensor) or tensor.dtype == target_dtype:
        return tensor
    # Mirror the DiT loader's one-tensor-at-a-time CPU-stream cast.  The VideoVAE source file is
    # large enough that a deferred all-at-once Metal cast can trip command-buffer limits on a busy
    # unified-memory system, while this route is a default-off load/decode experiment.
    with mx.stream(mx.cpu):
        tensor = tensor.astype(target_dtype)
        mx.eval(tensor)
    return tensor


def video_vae_parameter_dtype_summary(model) -> dict[str, object]:
    """Return a compact dtype/byte summary for loaded VideoVAE parameters."""

    by_dtype: dict[str, dict[str, int]] = {}
    by_region: dict[str, dict[str, int]] = {}
    total_params = 0
    total_bytes = 0
    floating_params = 0
    floating_bytes = 0
    for key, value in tree_flatten(model.parameters()):
        dtype = str(value.dtype)
        entry = by_dtype.setdefault(dtype, {"params": 0, "bytes": 0, "tensors": 0})
        entry["params"] += int(value.size)
        entry["bytes"] += int(value.nbytes)
        entry["tensors"] += 1
        region = key.split(".", 1)[0]
        region_entry = by_region.setdefault(region, {"params": 0, "bytes": 0, "tensors": 0})
        region_entry["params"] += int(value.size)
        region_entry["bytes"] += int(value.nbytes)
        region_entry["tensors"] += 1
        total_params += int(value.size)
        total_bytes += int(value.nbytes)
        if _is_float_tensor(value):
            floating_params += int(value.size)
            floating_bytes += int(value.nbytes)
    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_gb_decimal": total_bytes / 1e9,
        "floating_params": floating_params,
        "floating_bytes": floating_bytes,
        "floating_gb_decimal": floating_bytes / 1e9,
        "by_dtype": by_dtype,
        "by_region": by_region,
    }


def _video_vae_linear_is_quantizable(
    path: str,
    module: nn.Module,
    *,
    group_size: int = VIDEO_VAE_DECODER_QUANT_GROUP_SIZE,
) -> bool:
    """Return whether one VideoVAE decoder module is an MLX-quantizable ``Linear``."""

    return (
        path.startswith("decoder.")
        and isinstance(module, nn.Linear)
        and int(module.weight.shape[-1]) % int(group_size) == 0
    )


def _apply_video_vae_decoder_quantized_slots(
    model,
    *,
    bits: int,
    group_size: int = VIDEO_VAE_DECODER_QUANT_GROUP_SIZE,
    mode: str = VIDEO_VAE_DECODER_QUANT_MODE,
) -> set[str]:
    """Replace quantizable VideoVAE decoder linears with MLX QuantizedLinear slots.

    The replacement is structural only: meaningful packed weights are produced from the local source
    safetensors during loading.  ``decoder.x_embedder`` is intentionally skipped with the default
    group size because its input dimension is the 24-channel latent width, not a multiple of 64.
    """

    from .quantize import apply_quantized_slots

    quantized_paths: set[str] = set()

    def predicate(path: str, module: nn.Module):
        if _video_vae_linear_is_quantizable(path, module, group_size=group_size):
            quantized_paths.add(path)
            return {"group_size": int(group_size), "bits": int(bits), "mode": mode}
        return False

    apply_quantized_slots(model, predicate)
    return quantized_paths


def load_video_vae(
    model_dir: str | Path,
    strict: bool = True,
    decoder_quantization: VideoVAEDecoderQuantization | str | None = "off",
    precision: VideoVAEPrecision | str | None = "fp32",
):
    """Load the video VAE from a released ``video_vae/`` directory.

    The weights live in ``source/model.safetensors`` under the original CompVis-style names, which
    the port reproduces. Only the convolution weights move: torch stores
    ``(C_out, C_in, kD, kH, kW)`` and MLX wants ``(C_out, kD, kH, kW, C_in)``.

    ``decoder_quantization`` is a default-off deployment probe.  When set to ``"8bit"`` or
    ``"4bit"``, quantizable decoder ``Linear`` modules are replaced with MLX ``QuantizedLinear``
    slots and packed directly from the local source weights while loading.  Default ``"off"``
    preserves the historical unquantized VideoVAE path and parameter keys.

    ``precision`` is a separate default-off lower-precision load/decode probe.  Default ``"fp32"``
    preserves the source VideoVAE parameter dtypes.  ``"bf16"`` and ``"fp16"`` cast floating
    VideoVAE parameters while loading and mark the module so decode starts from matching
    latents/activations.
    """
    from .video_vae import VideoVAE, VideoVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)
    with open(model_dir / "source" / "config.json") as fh:
        source = json.load(fh)

    ch = source["ch"]
    config = VideoVAEConfig(
        in_channels=source["in_channels"],
        out_channels=source["out_ch"],
        latent_channels=source["z_channels"],
        block_out_channels=tuple(ch * m for m in source["ch_mult"]),
        layers_per_block=source["num_res_blocks"],
        spatial_downsample_factors=tuple(source["space_down"]),
        temporal_downsample_factors=tuple(source["time_down"]),
        decoder_num_layers=source["vit_decoder_kwargs"]["num_layers"],
        decoder_num_attention_heads=source["vit_decoder_kwargs"]["heads"],
        decoder_attention_head_dim=source["vit_decoder_kwargs"]["dim_head"],
        decoder_rope_theta=source["vit_decoder_kwargs"]["rope_theta"],
        decoder_rope_dim_ratio=source["vit_decoder_kwargs"]["rope_dim_ratio"],
        clip_length=wrapper.get("vae_clip_length", 17),
        token_drop=wrapper.get("vae_token_drop", 3),
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )
    quant_bits = _video_vae_decoder_quant_bits(decoder_quantization)
    precision_mode = _normalize_video_vae_precision(precision)
    model = VideoVAE(config)
    quantized_decoder_paths: set[str] = set()
    if quant_bits is not None:
        quantized_decoder_paths = _apply_video_vae_decoder_quantized_slots(model, bits=quant_bits)
    expected = {key for key, _ in tree_flatten(model.parameters())}

    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []
    raw_weights = dict(mx.load(str(model_dir / "source" / "model.safetensors")))
    for key in list(raw_weights):
        tensor = raw_weights.pop(key)
        # An all-zero buffer of the masked-autoencoding objective; the decoder never reads it.
        if key == "decoder.mask_token":
            del tensor
            continue
        module_path = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_path in quantized_decoder_paths:
            qweight, scales, *biases = mx.quantize(
                tensor,
                group_size=VIDEO_VAE_DECODER_QUANT_GROUP_SIZE,
                bits=int(quant_bits),
                mode=VIDEO_VAE_DECODER_QUANT_MODE,
            )
            weights[f"{module_path}.weight"] = qweight
            weights[f"{module_path}.scales"] = scales
            if biases:
                weights[f"{module_path}.biases"] = biases[0]
            mx.eval(qweight, scales, *biases)
            continue
        if key not in expected:
            unexpected.append(key)
            continue
        if tensor.ndim == 5:
            # Channels-last conv weights, materialized one at a time **on the CPU stream**.
            #
            # Deferring 10 GB of transposes into a single graph overruns the Metal command-buffer
            # deadline outright. Doing them individually on the GPU is enough on an idle machine but
            # still fails when something else is competing for the device — which is exactly when a
            # user is most likely to be loading a model. The CPU stream has no such deadline. It
            # trades some load time for not failing — a one-time cost on a path that otherwise
            # aborts a multi-hour run at the last component.
            with mx.stream(mx.cpu):
                tensor = mx.contiguous(tensor.transpose(0, 2, 3, 4, 1))
                mx.eval(tensor)
        tensor = _cast_video_vae_tensor(tensor, precision_mode)
        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Video VAE mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    model.update(tree_unflatten(list(weights.items())))
    if quant_bits is not None:
        model.video_vae_decoder_quantization = {
            "enabled": True,
            "bits": int(quant_bits),
            "group_size": VIDEO_VAE_DECODER_QUANT_GROUP_SIZE,
            "mode": VIDEO_VAE_DECODER_QUANT_MODE,
            "quantized_linear_count": len(quantized_decoder_paths),
            "quantized_linear_paths_sample": sorted(quantized_decoder_paths)[:8],
            "skipped_decoder_x_embedder_reason": "input dimension is not divisible by group_size=64",
        }
    else:
        model.video_vae_decoder_quantization = {"enabled": False}
    if hasattr(model, "set_decode_precision"):
        model.set_decode_precision(precision_mode)
    model.video_vae_precision = {
        "mode": precision_mode,
        "parameter_cast": precision_mode in {"bf16", "fp16"},
        "decode_input_dtype": {
            "bf16": "bfloat16",
            "fp16": "float16",
        }.get(precision_mode, "source_float32"),
    }
    mx.eval(model.parameters())
    model.video_vae_parameter_summary = video_vae_parameter_dtype_summary(model)
    model.video_vae_precision["parameter_summary"] = model.video_vae_parameter_summary
    return model


def read_audio_vae_config(model_dir: str | Path):
    """Read Audio VAE metadata without allocating model weights."""
    from .audio_vae import AudioVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json") as handle:
        kwargs = json.load(handle)["metadata"]["kwargs"]
    with open(model_dir / "config.json") as handle:
        wrapper = json.load(handle)
    return AudioVAEConfig(
        encoder_dim=kwargs["encoder_dim"],
        encoder_rates=tuple(kwargs["encoder_rates"]),
        latent_dim=kwargs["latent_dim"],
        latent_channels=kwargs["vae_latent_channels"],
        decoder_dim=kwargs["decoder_dim"],
        decoder_rates=tuple(kwargs["decoder_rates"]),
        sampling_rate=kwargs["sample_rate"],
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def load_audio_vae(model_dir: str | Path, strict: bool = True):
    """Load the audio VAE from a released ``audio_vae/`` directory.

    Weight norm is **folded** here: the checkpoint stores ``weight_g`` / ``weight_v`` and the
    effective weight is ``g * v / ||v||`` with the norm taken over every axis but the first. Folding
    once at load is exactly equivalent to recomputing it on every forward, and it lets the port hold
    a plain weight (proven equivalent by the parity test, which reconstructs the pair).
    """
    from .audio_vae import AudioVAE, AudioVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json") as fh:
        kwargs = json.load(fh)["metadata"]["kwargs"]
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)

    config = AudioVAEConfig(
        encoder_dim=kwargs["encoder_dim"],
        encoder_rates=tuple(kwargs["encoder_rates"]),
        latent_dim=kwargs["latent_dim"],
        latent_channels=kwargs["vae_latent_channels"],
        decoder_dim=kwargs["decoder_dim"],
        decoder_rates=tuple(kwargs["decoder_rates"]),
        sampling_rate=kwargs["sample_rate"],
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )
    model = AudioVAE(config)
    expected = {key for key, _ in tree_flatten(model.parameters())}

    raw = dict(mx.load(str(model_dir / "model.safetensors")))
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for key, tensor in raw.items():
        if key.endswith(".filter"):
            continue  # recomputed by kaiser_sinc_filter1d
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            g = raw[f"{base}_g"]
            v = tensor
            norm = mx.sqrt(mx.sum(mx.square(v.reshape(v.shape[0], -1)), axis=1)).reshape(-1, 1, 1)
            tensor = g * v / norm
            key = base
        elif key.endswith(".weight_g"):
            continue

        if key not in expected:
            unexpected.append(key)
            continue

        if key.endswith(".weight") and tensor.ndim == 3:
            # Transposed convs are stored (C_in, C_out, kL); plain convs (C_out, C_in, kL).
            tensor = tensor.transpose(1, 2, 0) if ".ups." in key else tensor.transpose(0, 2, 1)
        elif key.endswith(".alpha") and tensor.ndim == 3:
            tensor = tensor.transpose(0, 2, 1)  # (1, C, 1) -> (1, 1, C)

        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Audio VAE mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def parameter_summary(model: MiniMaxH3DiT) -> dict[str, object]:
    """Parameter counts and footprint, split by the AdaLN projections that can be dropped."""
    total = adaln = 0
    nbytes = adaln_bytes = 0
    for key, value in tree_flatten(model.parameters()):
        total += value.size
        nbytes += value.nbytes
        if ".adaln_proj." in key and key.startswith("blocks."):
            adaln += value.size
            adaln_bytes += value.nbytes
    return {
        "total_params": total,
        "adaln_params": adaln,
        "core_params": total - adaln,
        "total_gb": nbytes / 1e9,
        "adaln_gb": adaln_bytes / 1e9,
        "core_gb": (nbytes - adaln_bytes) / 1e9,
    }
