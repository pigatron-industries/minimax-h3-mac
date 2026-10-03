"""Fake-backed regressions for low-memory FL2VA staging and request geometry."""
from __future__ import annotations

import sys
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import minimax_h3_mlx.load as load_module
import minimax_h3_mlx.pipeline as pipeline_module
import minimax_h3_mlx.streaming as streaming_module
import minimax_h3_mlx.text_encoder as text_encoder_module
from minimax_h3_mlx.config import TAG_TEXT
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline


@dataclass
class VideoConfig:
    spatial_compression_ratio: int = 8
    latent_channels: int = 1
    latents_mean: tuple[float, ...] = (0.0,)
    latents_std: tuple[float, ...] = (1.0,)


@dataclass
class AudioConfig:
    latent_channels: int = 2
    sampling_rate: int = 32_000


@dataclass
class DitConfig:
    patch_size: tuple[int, int, int] = (1, 1, 1)
    num_layers: int = 1


class FakeScheduler:
    def __init__(self):
        self.timesteps = mx.array([1.0, 0.5], dtype=mx.float32)
        self.sigmas = mx.array([1.0, 0.5], dtype=mx.float32)

    def scale_noise(self, sample, timestep, noise):
        del timestep, noise
        return sample

    def step(self, model_output, timestep, sample):
        del model_output, timestep
        return sample + 1.0


@dataclass
class RuntimeHarness:
    events: list[str] = field(default_factory=list)
    video_inputs: list[np.ndarray] = field(default_factory=list)
    text_load_vision: list[bool] = field(default_factory=list)
    text_encoder_cls: type | None = field(init=False, default=None)
    _stack: ExitStack = field(init=False, default_factory=ExitStack)

    def __enter__(self):
        harness = self

        class FakeTextEncoder:
            def __init__(self, *args, **kwargs):
                del args
                harness.events.append("text-load")
                harness.text_load_vision.append(bool(kwargs.get("load_vision")))

            def encode(self, prompt, images=None):
                del prompt, images
                harness.events.append("text-encode")
                return (
                    mx.zeros((1, 1, 4), dtype=mx.bfloat16),
                    np.array([TAG_TEXT], dtype=np.int64),
                )

        class FakeDenoiser:
            blocks: list = []

            def __call__(self, video_rows, audio_rows, *args, **kwargs):
                del args, kwargs
                harness.video_inputs.append(
                    np.array(video_rows[0].astype(mx.float32), copy=True)
                )
                return mx.ones_like(video_rows), mx.ones_like(audio_rows)

        class FakeStreamingProvider:
            pass

        class FakeLoadedVae:
            pass

        def profiled(label, category, fn, **kwargs):
            del category, kwargs
            if label == "load.video_vae_low_memory":
                harness.events.append("video-vae-load")
            elif label == "load.streaming_transformer_low_memory":
                harness.events.append("dit-load")
            return fn()

        def fake_dit_loader(*args, **kwargs):
            del args, kwargs
            return FakeDenoiser(), FakeStreamingProvider()

        def fake_vae_loader(*args, **kwargs):
            del args, kwargs
            return FakeLoadedVae()

        def fake_schedules(self, num_steps):
            del self, num_steps
            return FakeScheduler(), FakeScheduler()

        self.text_encoder_cls = FakeTextEncoder
        self._stack.enter_context(
            patch.object(text_encoder_module, "MiniMaxH3TextEncoder", FakeTextEncoder)
        )
        self._stack.enter_context(patch.object(pipeline_module, "profiled_call", profiled))
        self._stack.enter_context(
            patch.object(streaming_module, "load_streaming_dit", fake_dit_loader)
        )
        self._stack.enter_context(patch.object(load_module, "load_video_vae", fake_vae_loader))
        self._stack.enter_context(patch.object(load_module, "load_audio_vae", fake_vae_loader))
        self._stack.enter_context(
            patch.object(MiniMaxH3Pipeline, "_build_schedules", fake_schedules)
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._stack.__exit__(exc_type, exc, traceback)

    def make_pipeline(self, checkpoint_root: Path) -> MiniMaxH3Pipeline:
        pipe = MiniMaxH3Pipeline(None, None, None, None)
        pipe._low_memory = True
        pipe._checkpoint_root = checkpoint_root
        pipe._text_encoder_path = checkpoint_root / "quantized-text"
        pipe._dit_path = checkpoint_root / "streaming-dit"
        pipe._video_config = VideoConfig()
        pipe._audio_config = AudioConfig()
        pipe._dit_config = DitConfig()
        pipe._block_provider = None
        pipe._ensure_cache = lambda *args, **kwargs: None
        pipe._decode_video = lambda *args, **kwargs: np.zeros(
            (124, 32, 32, 3), dtype=np.uint8
        )
        pipe._decode_audio = lambda *args, **kwargs: np.zeros((2, 1), dtype=np.float32)
        return pipe


def test_i2v_stages_share_prepared_pixels_and_keep_anchor_rows_fixed() -> None:
    with tempfile.TemporaryDirectory() as directory, RuntimeHarness() as harness:
        pipe = harness.make_pipeline(Path(directory))
        text_images: list[Image.Image] = []
        vae_images: list[Image.Image] = []

        class RecordingText(harness.text_encoder_cls):
            def encode(self, prompt, images=None):
                text_images.extend(images or [])
                return super().encode(prompt, images)

        def encode_keyframes(images, height, width):
            harness.events.append("keyframe-encode")
            vae_images.extend(images)
            assert (height, width) == (32, 32)
            return mx.arange(16, dtype=mx.float32).reshape(16, 1)

        original_release = pipe._release_component

        def release(name):
            if name == "text_encoder":
                harness.events.append("text-release")
            elif name == "video_vae":
                harness.events.append("video-vae-release")
            return original_release(name)

        original_seed = mx.random.seed

        def request_seed(seed):
            if seed == 0:
                harness.events.append("request-seed/noise")
            return original_seed(seed)

        pipe._encode_keyframes = encode_keyframes
        pipe._release_component = release
        image = Image.new("RGB", (32, 32), (12, 34, 56))
        with (
            patch.object(text_encoder_module, "MiniMaxH3TextEncoder", RecordingText),
            patch.object(mx.random, "seed", request_seed),
        ):
            result = pipe(
                "prompt",
                duration_seconds=5.0,
                num_inference_steps=2,
                images=[image],
                keyframe_anchors=("first",),
                height=32,
                width=32,
                verbose=False,
            )

        assert result.video.shape == (124, 32, 32, 3)
        assert harness.events[:8] == [
            "text-load",
            "text-encode",
            "text-release",
            "video-vae-load",
            "keyframe-encode",
            "video-vae-release",
            "dit-load",
            "request-seed/noise",
        ], harness.events
        assert text_images[0] is vae_images[0]
        assert np.array_equal(np.asarray(text_images[0]), np.asarray(vae_images[0]))
        assert len(harness.video_inputs) == 2
        assert np.array_equal(harness.video_inputs[0][:16], harness.video_inputs[1][:16])
        assert not np.array_equal(harness.video_inputs[0][16:], harness.video_inputs[1][16:])


def test_image_geometry_is_resolved_before_loading_models() -> None:
    with tempfile.TemporaryDirectory() as directory, RuntimeHarness() as harness:
        observed_sizes: list[tuple[int, int]] = []

        class GeometryProbe(harness.text_encoder_cls):
            def encode(self, prompt, images=None):
                del prompt
                observed_sizes.append(images[0].size)
                raise RuntimeError("geometry observed")

        pipe = harness.make_pipeline(Path(directory))
        with patch.object(text_encoder_module, "MiniMaxH3TextEncoder", GeometryProbe):
            try:
                pipe(
                    "prompt",
                    images=[Image.new("RGB", (4, 3), (1, 2, 3))],
                    keyframe_anchors=("first",),
                    verbose=False,
                )
            except RuntimeError as exc:
                assert str(exc) == "geometry observed"
            else:
                raise AssertionError("geometry probe should stop at text encoding")

        assert observed_sizes == [(1024, 768)]


def test_supported_anchor_forms_reach_packing_unchanged() -> None:
    with tempfile.TemporaryDirectory() as directory, RuntimeHarness() as harness:
        observed: list[tuple[str, ...]] = []
        original_build = pipeline_module.build_packed_sequence

        def record_layout(*args, **kwargs):
            anchors = args[6] if len(args) > 6 else kwargs["keyframe_anchors"]
            observed.append(tuple(anchors))
            return original_build(*args, **kwargs)

        with patch.object(pipeline_module, "build_packed_sequence", record_layout):
            for anchors in (("first",), ("last",), ("first", "last")):
                pipe = harness.make_pipeline(Path(directory))
                pipe._encode_keyframes = lambda images, height, width: mx.zeros(
                    (16 * len(images), 1), dtype=mx.float32
                )
                pipe(
                    "prompt",
                    images=[Image.new("RGB", (32, 32)) for _ in anchors],
                    keyframe_anchors=anchors,
                    height=32,
                    width=32,
                    num_inference_steps=1,
                    verbose=False,
                )

        assert observed == [("first",), ("last",), ("first", "last")]


def test_invalid_i2v_requests_fail_before_text_loading() -> None:
    with tempfile.TemporaryDirectory() as directory, RuntimeHarness() as harness:
        image = Image.new("RGB", (32, 32))

        def assert_rejected(**kwargs):
            before = len(harness.text_load_vision)
            try:
                harness.make_pipeline(Path(directory))("prompt", verbose=False, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid request unexpectedly succeeded: {kwargs}")
            assert len(harness.text_load_vision) == before

        assert_rejected(
            images=[image, image, image],
            keyframe_anchors=("first", "last", "first"),
        )
        assert_rejected(images=[image], keyframe_anchors=("middle",))
        assert_rejected(images=[image, image], keyframe_anchors=("last", "first"))
        assert_rejected(images=[image], keyframe_anchors=())
        assert_rejected(images=None, keyframe_anchors=("last",))
        assert_rejected(images=[image], keyframe_anchors=("first",), height=320)
        assert_rejected(images=[image], keyframe_anchors=("first",), height=31, width=32)


def test_t2v_skips_visual_loading_and_pre_dit_video_vae() -> None:
    with tempfile.TemporaryDirectory() as directory, RuntimeHarness() as harness:
        result = harness.make_pipeline(Path(directory))(
            "prompt", duration_seconds=5.0, num_inference_steps=1, verbose=False
        )

        assert result.video.shape == (124, 32, 32, 3)
        assert harness.text_load_vision == [False]
        assert harness.events.index("dit-load") < harness.events.index("video-vae-load")


def main() -> None:
    test_i2v_stages_share_prepared_pixels_and_keep_anchor_rows_fixed()
    test_image_geometry_is_resolved_before_loading_models()
    test_supported_anchor_forms_reach_packing_unchanged()
    test_invalid_i2v_requests_fail_before_text_loading()
    test_t2v_skips_visual_loading_and_pre_dit_video_vae()
    print("low-memory FL2VA staging, geometry, anchors, and T2V boundary passed")


if __name__ == "__main__":
    main()
