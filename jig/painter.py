"""The house-style book-art painter: image generation through AWS Bedrock.

Jig itself is a text LLM; this is the model call that paints. A ``PaintTool``-shaped capability is
deliberately *separate* from Jig's driving model - enabling or disabling it never touches the existing text
path. The painter generates one image and writes two derivatives (a 300 DPI print PNG and a compressed
EPUB variant) from that single generation, straight into the book's ``manuscript/images/`` folder.

Style is carried by *image-conditioned references*, not by prompt text alone. The locked house-style set
(three public-domain Rembrandt paintings - he died in 1669, so every one is public domain) is composited
into a single style plate and passed as an img2img conditioning image at low strength: the prompt names the
modern-tech subject, the plate supplies the chiaroscuro, the warm dark palette and the group-portrait
staging, and a whole book keeps one visual identity.

Vision stays off here. There is no self-review loop and no quality heuristic: the returned bytes are what
the model painted, offered for a human to curate before they ever go near a manuscript. ``render_derivatives``
only re-sizes and re-encodes the same bytes - it never judges them.
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .config import PainterConfig
from .errors import JigError

# The locked house-style reference set: public-domain Rembrandt paintings shipped with Jig under
# ``assets/painter/rembrandt``. Recorded provenance (each is PD because the artist died in 1669 and the
# museum image is released without restriction):
#   night-watch.jpg            Rembrandt, The Night Watch (1642), Rijksmuseum, Amsterdam — "File:The Night
#                              Watch - HD.jpg" on Wikimedia Commons, tooltip "Public domain".
#   drapers-guild.jpg          Rembrandt, The Syndics of the Drapers' Guild / De Staalmeesters (1662),
#                              Rijksmuseum — "File:Rembrandt - De Staalmeesters - The Syndics of the
#                              Clothmaker's Guild.jpg" on Commons, "Public domain".
#   self-portrait-two-circles.jpg  Rembrandt, Self-Portrait with Two Circles (c. 1665–69), Kenwood House —
#                              "File:Rembrandt Self-portrait (Kenwood).jpg" on Commons, "Public domain".
DEFAULT_REFERENCES = ("night-watch.jpg", "drapers-guild.jpg", "self-portrait-two-circles.jpg")

# A stable img2img strength that keeps the painting's technique without swallowing the subject.
DEFAULT_STRENGTH = 0.35

# The phrase the prompt template keeps constant across a book; the modern-tech subject is what varies.
STYLE_LOCK = ("in the style of Rembrandt's chiaroscuro group portraits: warm dark earth palette, deep "
              "brown-black shadows with single source of amber light, tenebrist candlelight modelling, "
              "thick impasto highlights, figures lit from one side against darkness, dramatic and painterly")

# Negative prompt shared with every generation: keep the scene painterly rather than photoreal-sketchy.
_NEGATIVE = ("flat colour, graphic design, logo, text, watermark, cartoon, line art, anime, photo, "
             "low detail, blurry, oversaturated")

SUPPORTED_MODELS = ("stability.sd3-5-large-v1:0", "amazon.titan-image-generator-v2:0")


class PainterError(JigError):
    """The painter is configured but could not generate (no AWS credentials, a refused request, ...)."""


def painter_assets_dir() -> Path:
    """The reference images shipped with Jig (``assets/painter/rembrandt`` next to this package's root)."""
    pkg = Path(__file__).resolve().parent
    # Repo checkout layout: <root>/assets/painter/rembrandt (package lives in <root>/jig/).
    repo = pkg.parent / "assets" / "painter" / "rembrandt"
    if repo.is_dir():
        return repo
    return pkg / "assets" / "painter" / "rembrandt"


def composite_reference_plate(images: list[bytes], *, strip_height: int = 512) -> bytes:
    """One style-conditioning image from the locked reference set.

    Each reference is scaled to ``strip_height`` pixels tall and laid out as a two-row grid (the first
    ``ceil(n/2)`` references on the top row, the remainder below), sized to the wider row and twice the strip
    height, so a whole generation carries the set's shared technique as a single img2img input (Bedrock takes
    one conditioning image per call; the composite is that one). The grid matters because Bedrock scales the
    conditioning image toward a one-megapixel target while preserving its ratio: a long single strip grows past
    1536 on the long axis and is rejected, while the roughly square two-row grid stays inside that limit.
    Returns PNG bytes.
    """
    if not images:
        raise PainterError("the reference set is empty; there is nothing to condition the painting on")
    cells = []
    for raw in images:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        scale = strip_height / im.height
        cells.append(im.resize((max(1, round(im.width * scale)), strip_height), Image.LANCZOS))
    top = cells[: (len(cells) + 1) // 2]
    bottom = cells[len(top):]
    width = max(sum(c.width for c in top), sum(c.width for c in bottom))
    plate = Image.new("RGB", (width, strip_height * 2), (0, 0, 0))
    for row, y in ((top, 0), (bottom, strip_height)):
        x = 0
        for cell in row:
            plate.paste(cell, (x, y))
            x += cell.width
    out = io.BytesIO()
    plate.save(out, "PNG", optimize=True)
    return out.getvalue()


class BedrockDriver:
    """Real AWS Bedrock calls (boto3, imported lazily). Kept behind the ``Painter`` so tests never touch AWS.

    Credentials come only from the environment variables the config names (or the AWS defaults); nothing is
    written anywhere, and this class never runs without them. ``generate`` is blocking, so callers run it in
    a worker thread.
    """

    def __init__(self, config: PainterConfig):
        self.config = config

    def credentials(self) -> dict[str, str]:
        ak = os.environ.get(self.config.access_key_id_env or "AWS_ACCESS_KEY_ID", "")
        sk = os.environ.get(self.config.access_key_secret_env or "AWS_SECRET_ACCESS_KEY", "")
        region = self.config.region or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION") or "us-east-1"
        missing = [name for name, value in (("access key id", ak), ("secret access key", sk)) if not value]
        if missing:
            raise PainterError("the painter is enabled but has no AWS credentials: the " + " and ".join(missing) +
                               " environment variable(s) are empty. Point [painter] access_key_id_env / "
                               "access_key_secret_env at the right names, or export the standard AWS_ACCESS_KEY_ID "
                               "and AWS_SECRET_ACCESS_KEY.")
        return {"aws_access_key_id": ak, "aws_secret_access_key": sk, "region_name": region}

    def generate(self, request: dict[str, Any], model_id: str) -> bytes:
        """Invoke Bedrock and return the raw painted image bytes."""
        import boto3  # optional dependency: only a real generation needs it

        creds = self.credentials()
        client = boto3.Session(**creds).client("bedrock-runtime", region_name=creds["region_name"])
        response = client.invoke_model(modelId=model_id, accept="application/json", contentType="application/json",
                                       body=json.dumps(request))
        body = json.loads(response["body"].read())
        # Stability: {"images": [base64]}; older Amazon frames: {"artifacts": [{"base64": ...}]}.
        if body.get("images"):
            return base64.b64decode(body["images"][0])
        if body.get("artifacts"):
            return base64.b64decode(body["artifacts"][0]["base64"])
        raise PainterError(f"the image model returned no image (response keys: {sorted(body)})")


class Painter:
    """Image generation for the book pipeline. Off (reports non-available) until ``[painter] enabled = true``.

    ``driver`` is the thing that turns a request into bytes: the real ``BedrockDriver`` by default, or a fake
    double in tests (no AWS credentials anywhere).
    """

    def __init__(self, config: PainterConfig, *, driver: Any | None = None, reference_dir: str = ""):
        self.config = config
        self.driver = driver if driver is not None else BedrockDriver(config)
        self.reference_dir: Path = self._resolve_reference_dir(reference_dir or config.reference_dir)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def _resolve_reference_dir(path: str) -> Path:
        if path:
            p = Path(path)
            return p if p.is_absolute() else p.resolve()
        return painter_assets_dir()

    def reference_names(self, requested: str | None) -> list[str]:
        """The locked set, or a single named override from it."""
        allowed = tuple(self.config.references) or DEFAULT_REFERENCES
        if requested:
            if requested not in allowed:
                raise PainterError(f"{requested!r} is not in the locked reference set {list(allowed)!r}")
            return [requested]
        return list(allowed)

    def reference_images(self, requested: str | None) -> list[bytes]:
        """Read the reference set (or one named reference) as bytes; absent files fail loudly, never silently."""
        out = []
        for name in self.reference_names(requested):
            path = self.reference_dir / name
            if not path.is_file():
                raise PainterError(f"reference image {name!r} is missing from {self.reference_dir}; the locked "
                                   f"reference set is part of the painter's build")
            out.append(path.read_bytes())
        return out

    def build_request(self, prompt: str, reference_images: list[bytes], *, aspect_ratio: str | None = None,
                      seed: int = 0, strength: float | None = None, negative_prompt: str = _NEGATIVE) -> dict[str, Any]:
        """The Bedrock request shape for the chosen provider, with the style plate as the conditioning image."""
        if not prompt or not prompt.strip():
            raise PainterError("a painting needs a subject prompt")
        plate = composite_reference_plate(reference_images)
        return {
            "prompt": prompt,
            "mode": "image-to-image",
            "image": base64.b64encode(plate).decode(),
            "strength": strength if strength is not None else self.config.strength,
            "output_format": self.config.output_format,
            "seed": int(seed),
            "negative_prompt": negative_prompt or _NEGATIVE,
        }

    def generate(self, prompt: str, *, reference_names: str | None = None, aspect_ratio: str | None = None,
                 seed: int = 0, strength: float | None = None) -> bytes:
        """Paint one image (the raw bytes). No review: what comes back is offered to a human, not judged."""
        if not self.enabled:
            raise PainterError("the painter is disabled; set [painter] enabled = true to use paint_image")
        request = self.build_request(prompt, self.reference_images(reference_names), aspect_ratio=aspect_ratio,
                                     seed=seed, strength=strength)
        return self.driver.generate(request, self.config.model_id)

    def render_derivatives(self, generated: bytes, slug: str, images_dir: Path) -> dict[str, Any]:
        """Two files from one generation, under ``manuscript/images/`` as the Book Drafting Pipeline ADR expects.

        ``{slug}_print.png`` is the full-resolution image written with 300 DPI metadata (Leanpub's print PDF
        sizes it from that, no resampling); ``{slug}_epub.jpg`` is the same pixels scaled to a reader-safe
        size and compressed for EPUB/MOBI. Both are derived from the single ``generated`` cache — nothing here
        looks at them or judges them.
        """
        images_dir = Path(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        im = Image.open(io.BytesIO(generated))
        im.load()
        dpi = self.config.print_dpi
        width_in, height_in = round(im.width / dpi, 2), round(im.height / dpi, 2)

        print_path = images_dir / f"{slug}_print.png"
        im.save(print_path, "PNG", dpi=(dpi, dpi), optimize=True)

        epub = im
        if im.width > self.config.epub_max_width:
            scale = self.config.epub_max_width / im.width
            epub = im.resize((self.config.epub_max_width, max(1, round(im.height * scale))), Image.LANCZOS)
        epub_path = images_dir / f"{slug}_epub.jpg"
        epub.convert("RGB").save(epub_path, "JPEG", quality=self.config.epub_quality, optimize=True,
                                 progressive=True)

        return {
            "generation": {"pixels": f"{im.width}x{im.height}", "print_size_in": f"{width_in} x {height_in}",
                           "dpi": dpi},
            "print": {"path": print_path.name, "pixels": f"{im.width}x{im.height}", "dpi": dpi,
                      "size_bytes": print_path.stat().st_size},
            "epub": {"path": epub_path.name, "pixels": f"{epub.width}x{epub.height}",
                     "size_bytes": epub_path.stat().st_size},
        }