"""MiniMax-H3's Qwen3-VL-32B conditioner, in MLX.

H3 does not use Qwen3-VL as a language model. It reads the **unnormalized** hidden state after the
50th of its 64 decoder layers (``hidden_states[50]``, where ``hidden_states[0]`` is the embedding
output) and feeds that straight into the DiT's ``condition_proj``. The language-model head, the
final norm and the last 14 decoder layers are never evaluated.

That is worth exploiting: the port loads **only the 50 layers it reads**, skipping ``lm_head``
(151936 x 5120) and layers 50-63 entirely. For a text-only request the vision tower is skipped too.

The transformer stack itself is mlx-vlm's ``qwen3_vl`` implementation — it already has the
interleaved M-RoPE, the ``mrope_section`` split and the deepstack visual merge — so this module only
supplies H3's request presentation, the truncated forward, and a loader that reads a subset.

**Request presentation** (from the reference; no chat template and no special tokens anywhere):
each keyframe contributes a ``"<Picture i>: "`` label followed by a vision block
(``<|vision_start|>``, one ``<|image_pad|>`` per merged patch, ``<|vision_end|>``), then the prompt
verbatim. The rows of a vision block are tagged **video**, not text — that tag is what the DiT's
AdaLN modulation keys off.
"""

from __future__ import annotations

import gc
import glob
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import TAG_TEXT, TAG_VIDEO
from .packing import TEXT_ENCODER_LAYER


class MiniMaxH3TextEncoder:
    """Qwen3-VL-32B truncated to the layers MiniMax-H3 actually conditions on."""

    def __init__(
        self,
        model_dir: str | Path,
        num_layers: int = TEXT_ENCODER_LAYER,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = True,
        verbose: bool = False,
        tokenizer_dir: str | Path | None = None,
        processor_dir: str | Path | None = None,
        stream_layers: bool = False,
        vision_model_dir: str | Path | None = None,
    ):
        from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer, Qwen3VLModel
        from mlx_vlm.models.qwen3_vl.vision import VisionModel

        model_dir = Path(model_dir)
        vision_model_dir = None if vision_model_dir is None else Path(vision_model_dir)
        with open(model_dir / "config.json") as fh:
            raw = json.load(fh)

        full_layers = raw["text_config"]["num_hidden_layers"]
        if full_layers <= num_layers:
            raise ValueError(
                f"MiniMax-H3 conditions on hidden_states[{num_layers}] of its Qwen3-VL conditioner, "
                f"which needs more than {num_layers} decoder layers, but the checkpoint has "
                f"{full_layers}. The last hidden state of a stack truncated to exactly {num_layers} "
                "layers is post-norm and is not the conditioning MiniMax-H3 expects."
            )

        if stream_layers and load_vision and vision_model_dir is None:
            raise ValueError(
                "vision_model_dir is required when stream_layers=True and load_vision=True; "
                "it must point to the indexed source checkpoint owning model.visual.* tensors."
            )

        self.num_layers = num_layers
        self.full_layers = full_layers
        self.dtype = dtype
        self.stream_layers = bool(stream_layers)

        text_raw = dict(raw["text_config"])
        # Resident mode builds the 50 evaluated layers. Streamed mode builds only one reusable
        # decoder-layer slot; its original BF16 weights are replaced before every layer forward.
        text_raw["num_hidden_layers"] = 1 if self.stream_layers else num_layers
        self.text_config = TextConfig.from_dict(text_raw)
        self.vision_config = VisionConfig.from_dict(raw["vision_config"])
        self.model_config = ModelConfig.from_dict(
            {
                **raw,
                "text_config": text_raw,
                "vision_config": raw["vision_config"],
                "model_type": raw.get("model_type", "qwen3_vl"),
            }
        )
        self.model_config.text_config = self.text_config
        self.model_config.vision_config = self.vision_config

        self.language = None if self.stream_layers else Qwen3VLModel(self.text_config)
        self._stream_layer = Qwen3VLDecoderLayer(self.text_config, layer_idx=0) if self.stream_layers else None
        self.vision = VisionModel(self.vision_config) if load_vision else None
        quant_path = model_dir / "quant_config.json"
        self.quantized = quant_path.exists()
        if self.quantized:
            import mlx.nn as nn
            from .quantize import apply_quantized_slots

            with quant_path.open() as handle:
                quant = json.load(handle)
            self.quant_config = quant

            def quantize_language(path, module):
                weight = getattr(module, "weight", None)
                should_quantize = (
                    hasattr(module, "to_quantized")
                    and isinstance(weight, mx.array)
                    and weight.ndim == 2
                    and weight.shape[-1] % int(quant["group_size"]) == 0
                )
                if not should_quantize:
                    return False
                return {
                    "group_size": int(quant["group_size"]),
                    "bits": int(quant["bits"]),
                    "mode": str(quant.get("mode", "affine")),
                }

            apply_quantized_slots(
                self._stream_layer if self.stream_layers else self.language,
                quantize_language,
            )
        if self.stream_layers:
            from .selective_loading import load_weight_map

            self._weight_map = load_weight_map(model_dir)
            self.skipped_tensors = len(self._weight_map) - sum(
                key == "model.language_model.embed_tokens.weight"
                or any(key.startswith(f"model.language_model.layers.{i}.") for i in range(num_layers))
                for key in self._weight_map
            )
            if self.vision is not None:
                self._load_stream_vision(vision_model_dir)
            if verbose:
                precision = "quantized" if self.quantized else "full-precision"
                print(f"  text encoder: {precision} weights, streaming {num_layers} layers")
        else:
            self.quant_config = None
            self._load_weights(model_dir, dtype, verbose)

        self.image_token_id = raw["image_token_id"]
        self.video_token_id = raw["video_token_id"]
        self.vision_start_token_id = raw["vision_start_token_id"]
        self.vision_end_token_id = raw["vision_end_token_id"]
        self.merge_size = self.vision_config.spatial_merge_size

        self._tokenizer = None
        self._image_processor = None
        self._model_dir = model_dir
        root = model_dir.parent
        self._tokenizer_dir = (
            Path(tokenizer_dir)
            if tokenizer_dir is not None
            else (root / "tokenizer" if (root / "tokenizer").exists() else model_dir)
        )
        self._processor_dir = (
            Path(processor_dir)
            if processor_dir is not None
            else (root / "processor" if (root / "processor").exists() else model_dir)
        )

    # -- loading ---------------------------------------------------------------------------

    def _wanted(self, key: str) -> str | None:
        """Map a checkpoint key onto this module's parameter path, or ``None`` to skip it."""
        if key.startswith("lm_head"):
            return None  # never evaluated
        if key.startswith("model.language_model."):
            rest = key[len("model.language_model.") :]
            if rest.startswith("layers."):
                index = int(rest.split(".")[1])
                if index >= self.num_layers:
                    return None  # beyond the conditioning layer
            # `norm` is loaded (it is 5120 floats) to keep the module tree complete, but it is never
            # applied: H3 reads the hidden state *before* the final norm.
            return ("language", rest)
        if key.startswith("model.visual."):
            if self.vision is None:
                return None
            return ("vision", key[len("model.visual.") :])
        return None

    def _load_weights(self, model_dir: Path, dtype: mx.Dtype, verbose: bool) -> None:
        from mlx.utils import tree_flatten, tree_unflatten

        shards = sorted(glob.glob(str(model_dir / "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"No safetensors in {model_dir}.")

        expected = {
            "language": {k for k, _ in tree_flatten(self.language.parameters())},
            "vision": set() if self.vision is None else {k for k, _ in tree_flatten(self.vision.parameters())},
        }
        remaining = {bucket: set(keys) for bucket, keys in expected.items()}
        loaded = 0
        skipped = 0
        for shard in shards:
            updates: dict[str, list[tuple[str, mx.array]]] = {"language": [], "vision": []}
            for key, tensor in mx.load(shard).items():
                target = self._wanted(key)
                if target is None:
                    skipped += 1
                    continue
                bucket, path = target
                if path not in expected[bucket]:
                    skipped += 1
                    continue
                updates[bucket].append(
                    (path, tensor if self.quantized else tensor.astype(dtype))
                )
                remaining[bucket].discard(path)
                loaded += 1

            for bucket, module in (("language", self.language), ("vision", self.vision)):
                if module is None or not updates[bucket]:
                    continue
                update_items = updates[bucket]
                if bucket == "vision":
                    update_items = list(self.vision.sanitize(dict(update_items)).items())
                module.update(tree_unflatten(update_items))
                mx.eval(*(tensor for _, tensor in update_items))
            if verbose:
                print(f"  {Path(shard).name}: {loaded} tensors loaded")

        for bucket in ("language", "vision"):
            missing = sorted(remaining[bucket])
            if missing:
                raise KeyError(
                    f"{bucket} encoder missing {len(missing)} tensors, e.g. {missing[:4]}."
                )
        self.skipped_tensors = skipped

    def _load_stream_vision(self, vision_model_dir: Path | None) -> None:
        """Load exactly the visual tensors from the indexed upstream source checkpoint."""

        from mlx.utils import tree_flatten, tree_unflatten

        from .selective_loading import load_selected_mlx_tensors, load_weight_map

        if vision_model_dir is None:
            # The constructor checks this combination before constructing the module. Keep the
            # guard here as well so this helper cannot ever silently choose the language directory.
            raise ValueError("vision_model_dir is required for streamed vision loading")
        index_path = vision_model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"visual checkpoint index not found at {index_path}")

        indexed = load_weight_map(vision_model_dir)
        prefix = "model.visual."
        selected = {key[len(prefix) :] for key in indexed if key.startswith(prefix)}
        expected = {key for key, _ in tree_flatten(self.vision.parameters())}
        missing = sorted(expected - selected)
        unexpected = sorted(selected - expected)
        if missing or unexpected:
            raise KeyError(
                f"visual tensor set mismatch in vision_model_dir={vision_model_dir}: "
                f"expected {len(expected)}, selected {len(selected)}, missing {len(missing)} "
                f"{missing[:4]}, unexpected {len(unexpected)} {unexpected[:4]}"
            )

        source_keys = [prefix + key for key in sorted(expected)]
        loaded = load_selected_mlx_tensors(vision_model_dir, source_keys)
        stripped = {key[len(prefix) :]: value for key, value in loaded.items()}
        sanitized = self.vision.sanitize(stripped)
        sanitized_keys = set(sanitized)
        if sanitized_keys != expected:
            missing_after = sorted(expected - sanitized_keys)
            unexpected_after = sorted(sanitized_keys - expected)
            raise KeyError(
                f"sanitized visual tensor set mismatch in vision_model_dir={vision_model_dir}: "
                f"expected {len(expected)}, got {len(sanitized_keys)}, missing {missing_after[:4]}, "
                f"unexpected {unexpected_after[:4]}"
            )
        self.vision.update(tree_unflatten(sorted(sanitized.items())))
        mx.eval(self.vision.parameters())

    # -- tokenizer / processor -------------------------------------------------------------

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(self._tokenizer_dir))
            required_token_id = max(
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            )
            if len(tokenizer) <= required_token_id:
                raise ValueError(
                    f"Tokenizer at {self._tokenizer_dir} has only {len(tokenizer)} tokens, "
                    f"but the checkpoint requires token id {required_token_id}. "
                    "Pass the source FL2VA tokenizer directory via `tokenizer_dir`."
                )
            self._tokenizer = tokenizer
        return self._tokenizer

    @property
    def image_processor(self):
        """Load the image-only Qwen2-VL processor without its torch-backed video sibling."""

        if self._image_processor is None:
            try:
                from transformers.models.qwen2_vl import Qwen2VLImageProcessorPil as ImageProcessor
            except ImportError:  # transformers 4.x exposes the same NumPy/Pillow path without the suffix.
                from transformers.models.qwen2_vl import Qwen2VLImageProcessor as ImageProcessor

            self._image_processor = ImageProcessor.from_pretrained(str(self._processor_dir))
        return self._image_processor

    # -- request presentation --------------------------------------------------------------

    def build_request(self, prompt: str, images: list | None = None):
        """Build H3's token sequence and its per-row modality tags.

        Returns ``(input_ids, token_tags, vision_inputs)``; ``vision_inputs`` is ``None`` for a
        text-only request, otherwise the processor's ``pixel_values`` / ``image_grid_thw``.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be a single string, got {type(prompt).__name__}.")

        token_ids: list[int] = []
        token_tags: list[int] = []
        vision_inputs = None

        if images:
            vision = self.image_processor(images=images, return_tensors="np")
            pixel_values = np.asarray(vision["pixel_values"])
            grid_thw = np.asarray(vision["image_grid_thw"])
            merge = self.image_processor.merge_size**2
            expected_rows = 0
            for index in range(len(images)):
                grid = np.asarray(grid_thw[index], dtype=np.int64)
                patch_rows = int(np.prod(grid))
                if patch_rows % merge:
                    raise ValueError(
                        f"image[{index}] grid rows are not divisible by merge area: "
                        f"observed {patch_rows}, expected a multiple of {merge}"
                    )
                expected_rows += patch_rows
            if pixel_values.shape[0] != expected_rows:
                raise ValueError(
                    f"visual rows/image-grid mismatch: observed {pixel_values.shape[0]} visual rows, "
                    f"expected {expected_rows} from image_grid_thw"
                )
            start = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
            pad = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            end = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

            for index in range(len(images)):
                num_image_tokens = int(grid_thw[index].prod()) // merge
                label_ids = self.tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = [start] + [pad] * num_image_tokens + [end]
                token_ids += label_ids + vision_ids
                # The whole vision block is tagged *video*; only the label stays text.
                token_tags += [TAG_TEXT] * len(label_ids) + [TAG_VIDEO] * len(vision_ids)
            vision_inputs = (pixel_values, grid_thw)

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [TAG_TEXT] * len(prompt_ids)
        if not token_ids:
            raise ValueError(
                "The request produced no input token IDs. Check `tokenizer_dir` and provide "
                "a prompt that tokenizes to at least one token."
            )

        return (
            mx.array(np.array([token_ids], dtype=np.int32)),
            np.array(token_tags, dtype=np.int64),
            vision_inputs,
        )

    # -- forward ---------------------------------------------------------------------------

    def _load_stream_tensor(self, key: str) -> mx.array:
        from .selective_loading import load_selected_mlx_tensors

        if key not in self._weight_map:
            raise KeyError(f"text encoder is missing streamed tensor {key!r}")
        tensor = load_selected_mlx_tensors(self._model_dir, [key])[key]
        return tensor if self.quantized else tensor.astype(self.dtype)

    def _load_stream_layer(self, layer_idx: int) -> None:
        from mlx.utils import tree_flatten, tree_unflatten
        from .selective_loading import load_selected_mlx_tensors

        layer = self._stream_layer
        expected = {key for key, _ in tree_flatten(layer.parameters())}
        # The previous layer's forward has already been evaluated. Remove its arrays before reading
        # the next layer so peak residency is one layer rather than old+new during the handoff.
        layer.update(tree_unflatten([(key, mx.array(0, dtype=mx.uint32)) for key in expected]))
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if clear_cache is not None:
            clear_cache()
        prefix = f"model.language_model.layers.{layer_idx}."
        source_by_target = {
            key[len(prefix) :]: key for key in self._weight_map if key.startswith(prefix)
        }
        missing = sorted(expected - source_by_target.keys())
        if missing:
            raise KeyError(f"text encoder layer {layer_idx} is missing tensors, e.g. {missing[:4]}")
        loaded = load_selected_mlx_tensors(
            self._model_dir,
            [source_by_target[key] for key in sorted(expected)],
        )
        updates = [
            (
                key,
                loaded[source_by_target[key]]
                if self.quantized
                else loaded[source_by_target[key]].astype(self.dtype),
            )
            for key in sorted(expected)
        ]
        layer.update(tree_unflatten(updates))
        mx.eval(*(tensor for _, tensor in updates))

    def _hidden_states(
        self,
        input_ids: mx.array,
        position_ids: mx.array,
        inputs_embeds: mx.array | None = None,
        visual_pos_masks: mx.array | None = None,
        deepstack_visual_embeds: list | None = None,
        visual_embeds: mx.array | None = None,
    ) -> mx.array:
        """Run the truncated stack and return the hidden state **before** the final norm."""
        from mlx_vlm.models.base import create_attention_mask

        visual_positions = None
        if self.stream_layers and visual_pos_masks is not None:
            if visual_pos_masks.ndim != 2 or visual_pos_masks.shape[0] != 1:
                raise ValueError(
                    "streamed text encoding accepts exactly one request at a time; "
                    f"got visual mask shape {visual_pos_masks.shape}"
                )
            mask_np = np.asarray(visual_pos_masks[0], dtype=bool)
            visual_positions = mx.array(np.flatnonzero(mask_np), dtype=mx.uint32)
            if visual_embeds is not None and visual_embeds.shape[0] != len(visual_positions):
                raise ValueError(
                    f"visual rows/image-pad mismatch: observed {visual_embeds.shape[0]} visual rows, "
                    f"expected {len(visual_positions)} image-pad positions"
                )

        def replace_visual_rows(hidden: mx.array, values: mx.array | None) -> mx.array:
            if values is None:
                return hidden
            if visual_pos_masks is None:
                raise ValueError("visual embeddings were supplied without image-pad positions")
            from mlx_vlm.models.qwen3_vl.qwen3_vl import Model

            merged, _ = Model.merge_input_ids_with_image_features(
                values.astype(hidden.dtype),
                hidden,
                input_ids,
                self.image_token_id,
                self.video_token_id,
            )
            return merged

        def add_deepstack_rows(hidden: mx.array, values: mx.array) -> mx.array:
            if visual_positions is None:
                raise ValueError("deep-stack visual embeddings were supplied without image-pad positions")
            if values.shape[0] != len(visual_positions):
                raise ValueError(
                    f"deep-stack visual rows/image-pad mismatch: observed {values.shape[0]} rows, "
                    f"expected {len(visual_positions)} image-pad positions"
                )
            row = hidden[0]
            row = row.at[visual_positions].add(values.astype(row.dtype))
            return mx.expand_dims(row, axis=0)

        if self.stream_layers:
            embedding_key = "model.language_model.embed_tokens.weight"
            embedding = self._load_stream_tensor(embedding_key)
            if self.quantized:
                from .selective_loading import load_selected_mlx_tensors

                stem = embedding_key[: -len("weight")]
                aux_keys = [stem + "scales", stem + "biases"]
                aux = load_selected_mlx_tensors(self._model_dir, aux_keys)
                # QuantizedEmbedding cannot be used as the streamed layer slot. Select the prompt
                # rows while they are still packed, then dequantize only those rows; dequantizing
                # the complete 151936 x 5120 table would defeat low-memory text streaming.
                h = mx.dequantize(
                    embedding[input_ids],
                    aux[stem + "scales"][input_ids],
                    aux[stem + "biases"][input_ids],
                    group_size=int(self.quant_config["group_size"]),
                    bits=int(self.quant_config["bits"]),
                    mode=str(self.quant_config.get("mode", "affine")),
                )
                del aux
            else:
                h = embedding[input_ids]
            mx.eval(h)
            del embedding
            gc.collect()
            clear_cache = getattr(mx, "clear_cache", None)
            if clear_cache is not None:
                clear_cache()
            mask = create_attention_mask(h, None)
            h = replace_visual_rows(h, visual_embeds)
            layer = self._stream_layer
            position_embeddings = None
            if position_ids is not None and not layer.self_attn.rotary_emb.fused_apply:
                position_embeddings = layer.self_attn.rotary_emb(h, position_ids)
            for layer_idx in range(self.num_layers):
                self._load_stream_layer(layer_idx)
                h = layer(h, mask, None, position_ids, position_embeddings)
                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    h = add_deepstack_rows(h, deepstack_visual_embeds[layer_idx])
                # Materialize before replacing this slot with the next layer's weights, ensuring
                # that at most one full decoder layer is resident.
                mx.eval(h)
                gc.collect()
            return h

        model = self.language
        h = model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        h = replace_visual_rows(h, visual_embeds)
        mask = create_attention_mask(h, None)

        position_embeddings = None
        if position_ids is not None and not model.layers[0].self_attn.rotary_emb.fused_apply:
            position_embeddings = model.layers[0].self_attn.rotary_emb(h, position_ids)

        for layer_idx, layer in enumerate(model.layers):
            h = layer(h, mask, None, position_ids, position_embeddings)
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                h = model._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
        # No `model.norm(h)`: H3 conditions on the unnormalized state.
        return h

    def encode(self, prompt: str, images: list | None = None) -> tuple[mx.array, np.ndarray]:
        """Encode a request into ``((1, num_text_tokens, 5120), (num_text_tokens,))``."""
        from mlx_vlm.models.qwen3_vl.language import LanguageModel

        input_ids, token_tags, vision_inputs = self.build_request(prompt, images)

        inputs_embeds = None
        visual_pos_masks = None
        deepstack_embeds = None
        grid_thw = None
        visual_embeds = None

        if vision_inputs is not None:
            if self.vision is None:
                raise ValueError("This encoder was built with `load_vision=False`; it cannot take images.")
            pixel_values, grid_np = vision_inputs
            grid_thw = mx.array(grid_np.astype(np.int32))
            hidden, deepstack_embeds = self.vision(
                mx.array(pixel_values).astype(self.dtype), grid_thw, output_hidden_states=True
            )
            image_mask = input_ids == self.image_token_id
            visual_pos_masks = image_mask
            image_rows = int(np.asarray(image_mask).sum())
            if hidden.shape[0] != image_rows:
                raise ValueError(
                    f"visual rows/image-pad mismatch: observed {hidden.shape[0]} visual rows, "
                    f"expected {image_rows} image-pad positions"
                )
            if deepstack_embeds is None:
                deepstack_embeds = []
            for index, deepstack in enumerate(deepstack_embeds):
                if deepstack.shape[0] != image_rows:
                    raise ValueError(
                        f"deep-stack visual rows/image-pad mismatch at index {index}: observed "
                        f"{deepstack.shape[0]} rows, expected {image_rows} image-pad positions"
                    )
            visual_embeds = hidden.astype(self.dtype)
            deepstack_embeds = [value.astype(self.dtype) for value in deepstack_embeds]
            if self.stream_layers:
                def detach_visual(array: mx.array) -> mx.array:
                    if self.dtype == mx.bfloat16:
                        host = np.array(array.view(mx.uint16), copy=True)
                        detached = mx.array(host, dtype=mx.uint16).view(mx.bfloat16)
                    else:
                        host = np.array(array.astype(self.dtype), copy=True)
                        detached = mx.array(host).astype(self.dtype)
                    mx.eval(detached)
                    return detached

                visual_embeds = detach_visual(visual_embeds)
                deepstack_embeds = [detach_visual(value) for value in deepstack_embeds]
                self.vision = None
                del hidden, pixel_values
                gc.collect()
                clear_cache = getattr(mx, "clear_cache", None)
                if clear_cache is not None:
                    clear_cache()

        # Qwen3-VL's 3D M-RoPE index, derived from the vision-start/pad token ids.
        position_ids, _ = LanguageModel.get_rope_index(
            self, input_ids, image_grid_thw=grid_thw, video_grid_thw=None, attention_mask=None
        )

        hidden_states = self._hidden_states(
            input_ids,
            position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_embeds,
            visual_embeds=visual_embeds,
        )
        mx.eval(hidden_states)
        return hidden_states, token_tags

    # `LanguageModel.get_rope_index` reads `self.config`; expose the same attribute.
    @property
    def config(self):
        return self.model_config
