"""The painter: tool-call shape and the two-derivative output contract, offline.

Nothing here needs AWS, boto3 or a model server: the driver behind ``Painter`` is a deterministic fake that
paints a solid-colour PNG from the seed, exactly the way the real ``BedrockDriver`` is swappable on the live
agent. The reference set, the gate and the sandbox are all real (this machine's repo assets, a real SQLite
database, a real directory jail).

The parts of this capability that genuinely cannot run here - a real Bedrock generation and a real Leanpub
preview build - are not faked into passing: the fake driver proves the tool-call shape and the file contract,
and the report says plainly that the live generation and the preview gate are the human's next step.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from jig.config import PainterConfig, load_config
from jig.constants import Decision, Effect, Mode, ToolCategory
from jig.errors import ToolArgumentError, ToolError
from jig.painter import DEFAULT_REFERENCES, Painter, painter_assets_dir
from jig.policy.gate import CallContext
from jig.sandbox import Sandbox
from jig.tools.builtin import build_registry
from jig.tools.painter import register_painter_tools
from jig.tools.registry import ToolContext

WIDTH, HEIGHT = 1536, 1024  # what a real SD3.5 Large generation would look like


class FakeDriver:
    """Deterministic stand-in for Bedrock: paints a solid PNG, varying with the seed, nothing else."""

    def __init__(self, size: tuple[int, int] = (WIDTH, HEIGHT)):
        self.size = size
        self.calls: list[tuple[dict, str]] = []

    def generate(self, request: dict, model_id: str) -> bytes:
        self.calls.append((request, model_id))
        shade = 40 + (int(request["seed"]) % 6) * 30
        png = io.BytesIO()
        Image.new("RGB", self.size, (shade, shade // 2, shade // 4)).save(png, "PNG")
        return png.getvalue()


def _painter(tmp_path: Path, **overrides) -> Painter:
    cfg = PainterConfig(enabled=True, reference_dir=str(painter_assets_dir()), **overrides)
    return Painter(cfg, driver=FakeDriver())


def _tool_spec(painter: Painter):
    registry = build_registry()
    register_painter_tools(registry, painter)
    return registry.get("paint_image")


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(sandbox=Sandbox(tmp_path / "sandbox", "default"), memory=None, store=None,
                       config=load_config(), http=None, mode=Mode.ACTION, run_id="r-painter", task_id=None)


# The reference set -------------------------------------------------------------


def test_the_locked_references_are_public_domain_and_ship_with_jig():
    """The style images are the fixed house set, they exist next to the package, and provenance is recorded."""
    assert DEFAULT_REFERENCES == ("night-watch.jpg", "drapers-guild.jpg", "self-portrait-two-circles.jpg")
    for name in DEFAULT_REFERENCES:
        assert (painter_assets_dir() / name).is_file(), f"{name} must ship in assets/painter/rembrandt"
    source = Path(Painter.__module__ + ".py")
    text = (Path(__file__).parent.parent / "jig" / "painter.py").read_text(encoding="utf-8")
    assert "Public domain" in text and "died in 1669" in text, "why the set is public domain is part of the contract"


# Tool-call shape through the registry ------------------------------------------


def test_paint_image_is_an_ask_gated_side_effect_tool(tmp_path):
    painter = _painter(tmp_path)
    spec = _tool_spec(painter)
    assert spec.name == "paint_image"
    assert spec.effect == Effect.SIDE_EFFECT and spec.category == ToolCategory.FILES
    assert spec.default_decision == Decision.ASK, "painting spends an image call and writes files: ask first"
    assert spec.is_available(), "offered while [painter] enabled = true"
    props = spec.parameters["properties"]
    assert set(props) == {"prompt", "slug", "manuscript_images_dir", "reference", "aspect_ratio", "seed",
                          "strength"}
    assert props["slug"]["description"] and props["prompt"]["description"]
    assert "curate" in spec.description.lower() or "human" in spec.description.lower(), \
        "the model is told the output is not self-reviewed"


def test_the_tool_is_not_offered_when_the_painter_is_disabled(tmp_path):
    disabled = Painter(PainterConfig(enabled=False), driver=FakeDriver())
    registry = build_registry()
    register_painter_tools(registry, disabled)
    assert "paint_image" not in [t.name for t in registry.available()], \
        "no painter configured: the tool is not offered to the model and nothing else changes"
    assert registry.get("paint_image").is_available() is False


async def test_a_bad_slug_is_refused_before_any_generation(tmp_path):
    painter = _painter(tmp_path)
    with pytest.raises(ToolArgumentError, match="slug"):
        await _tool_spec(painter).fn(_context(tmp_path), "a server room", "../escape")


# The one-generation, two-derivative output contract -----------------------------


def test_build_request_carries_the_style_plate_as_conditioning(tmp_path):
    painter = _painter(tmp_path)
    refs = painter.reference_images(None)
    request = painter.build_request("A rack of servers lit by candlelight", refs, seed=42,
                                    aspect_ratio="16:9", strength=0.3)
    assert request["mode"] == "image-to-image"
    assert request["seed"] == 42 and request["strength"] == 0.3
    assert request["prompt"] == "A rack of servers lit by candlelight"
    assert request["output_format"] == "png"
    assert len(request["image"]) > 0
    plate = io.BytesIO(base64.b64decode(request["image"]))
    assert Image.open(plate).size[1] == 512, "the composite style plate is a strip of the reference set"
    assert request["negative_prompt"], "there is a negative prompt with every request"


def test_two_derivatives_from_one_generation(tmp_path):
    painter = _painter(tmp_path)
    generated = FakeDriver().generate({"seed": 1}, "any")
    images_dir = tmp_path / "manuscript" / "images"
    result = painter.render_derivatives(generated, "rack-room", images_dir)

    print_path = images_dir / "rack-room_print.png"
    epub_path = images_dir / "rack-room_epub.jpg"
    assert print_path.is_file() and epub_path.is_file()

    print_im = Image.open(print_path)
    assert print_im.format == "PNG" and print_im.size == (WIDTH, HEIGHT)
    px, py = print_im.info.get("dpi") or (0, 0)
    assert abs(px - painter.config.print_dpi) < 0.01 and abs(py - painter.config.print_dpi) < 0.01, \
        "the print PNG carries ~300 DPI metadata so Leanpub's print PDF sizes it without resampling"

    epub_im = Image.open(epub_path)
    assert epub_im.format == "JPEG" and epub_im.mode == "RGB"
    assert epub_im.width == painter.config.epub_max_width, "the EPUB variant is scaled to a reader-safe width"
    assert epub_im.height < HEIGHT
    assert epub_path.stat().st_size < print_path.stat().st_size, "the EPUB variant is the compressed one"

    assert result["generation"]["dpi"] == 300
    assert result["print"]["path"].endswith("_print.png") and result["epub"]["path"].endswith("_epub.jpg")
    assert result["print"]["pixels"] == f"{WIDTH}x{HEIGHT}" and result["epub"]["pixels"] == f"{epub_im.width}x{epub_im.height}"


def test_the_whole_tool_call_writes_both_files_into_the_manuscript(tmp_path):
    painter = _painter(tmp_path)
    spec = _tool_spec(painter)
    ctx = _context(tmp_path)
    result = asyncio.run(spec.fn(ctx, "A dark server room", "rack-room"))

    rel = result["images_dir"]
    assert rel == "manuscript/images"
    assert (ctx.sandbox.resolve(rel) / "rack-room_print.png").is_file()
    assert (ctx.sandbox.resolve(rel) / "rack-room_epub.jpg").is_file()
    assert ctx.sandbox.resolve(result["print"]["path"]).is_file()
    assert ctx.sandbox.resolve(result["epub"]["path"]).is_file()
    assert "curate" in result["note"].lower() and "Leanpub" in result["note"]
    assert painter.driver.calls and painter.driver.calls[0][1] == "stability.sd3-5-large-v1:0", \
        "the configured Bedrock model is the one the driver is asked to invoke"


def test_reference_conditioning_is_deterministic_for_a_given_seed(tmp_path):
    painter = _painter(tmp_path)
    a = painter.reference_images(None)
    b = painter.reference_images(None)
    req_a = painter.build_request("same", a, seed=7)
    req_b = painter.build_request("same", b, seed=7)
    assert req_a["image"] == req_b["image"], "same references, same seed: same conditioning, same retry space"
    changed = painter.build_request("same", a, seed=8)
    assert changed["seed"] == 8, "a different seed changes the random pick but not the style plate"


def test_every_bound_reference_is_read_or_fails_loudly(tmp_path):
    painter = _painter(tmp_path)
    assert len(painter.reference_images(None)) == len(painter.reference_names(None)) == 3
    single = painter.reference_images("night-watch.jpg")
    assert len(single) == 1
    with pytest.raises(Exception, match="locked reference set"):
        painter.reference_names("mona-lisa.jpg")


def test_painter_config_rejects_unknown_models_and_bad_strength(tmp_path):
    from jig.config import ConfigError, load_config

    (tmp_path / "jig.toml").write_text(
        "[model]\nbase_url = 'http://127.0.0.1:8080/v1'\n"
        "[painter]\nenabled = true\nmodel_id = 'some.other-model:0'\n")
    with pytest.raises(ConfigError, match="model_id"):
        load_config(str(tmp_path / "jig.toml"))
    (tmp_path / "jig2.toml").write_text(
        "[model]\nbase_url = 'http://127.0.0.1:8080/v1'\n"
        "[painter]\nenabled = true\nstrength = 3.0\n")
    with pytest.raises(ConfigError, match="strength"):
        load_config(str(tmp_path / "jig2.toml"))