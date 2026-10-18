"""The painter tool: one Bedrock image generation, written as two derivatives into the manuscript.

Registered whenever Jig starts, exactly like the connector tools: the tool is *offered* to the model only
while ``[painter] enabled = true``, so a Jig with no painter configured behaves today the same as it always
did - the tool is simply not offered, and nothing else changes. Curating is a human's job: this tool never
looks at what it wrote, so there is no self-review loop to leak vision in by the back door.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ..constants import Decision, Effect, ToolCategory
from ..errors import ToolArgumentError, ToolError
from ..painter import Painter
from .registry import ToolContext, ToolRegistry

DEFAULT_IMAGES_DIR = "manuscript/images"
# A slug becomes file names; keep it to a safe stem so it can never leave the images folder.
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


def register_painter_tools(registry: ToolRegistry, painter: Painter) -> None:
    tool = registry.tool

    @tool(
        description="Paint one book illustration with the house-style painter (AWS Bedrock, a Rembrandt look "
        "carried by a locked public-domain reference set instead of prompt words). Give the modern-tech subject "
        "concretely - a server room, a cloud, an MCP handshake, a computer by candlelight. One generation is "
        "written as two files under manuscript/images/: a 300 DPI print PNG and a compressed EPUB JPEG. This "
        "is an action: it spends an image-model call and writes files. The result is NOT reviewed - a human "
        "must look at every image before it goes into a manuscript.",
        effect=Effect.SIDE_EFFECT,
        category=ToolCategory.FILES,
        default_decision=Decision.ASK,
        available=lambda: painter.enabled,
        args={
            "prompt": "The scene to paint, described concretely (the Rembrandt style lock comes from the "
                      "reference set, so keep this about the subject: what is in the scene, what is happening, "
                      "the light).",
            "slug": "File stem for the two output files, lower-case letters, digits, '-' or '_' (for example "
                     "'rack-room-nightshift').",
            "manuscript_images_dir": "Folder relative to the workspace root where the images land, default "
                                     "manuscript/images (the Leanpub tree the book pipeline already uses).",
            "reference": "One reference image from the locked set to use alone ('' uses the whole set).",
            "aspect_ratio": "Requested frame: 16:9, 1:1, 3:2, 4:5 or 9:16 (default: the configured one).",
            "seed": "0 for a fresh random pick; a fixed seed to regenerate from the same composition.",
            "strength": "How strongly the style plate shows through (0-1); leave empty for the standard 0.35.",
        },
    )
    async def paint_image(ctx: ToolContext, prompt: str, slug: str, manuscript_images_dir: str = DEFAULT_IMAGES_DIR,
                          reference: str = "", aspect_ratio: str = "", seed: int = 0,
                          strength: float | None = None) -> dict[str, Any]:
        if not painter.enabled:
            raise ToolError("paint_image is not available: the painter is disabled")
        if not _SLUG.fullmatch(slug):
            raise ToolArgumentError("paint_image: slug must be 2-64 characters of lower-case letters, digits, "
                                    "'-' or '_' (it becomes file names)")
        generated = await asyncio.to_thread(
            painter.generate, prompt, reference_names=reference or None,
            aspect_ratio=aspect_ratio or None, seed=seed, strength=strength)
        target = ctx.sandbox.resolve(manuscript_images_dir)
        result = await asyncio.to_thread(painter.render_derivatives, generated, slug, target)
        result["slug"] = slug
        result["images_dir"] = ctx.sandbox.relative(target)
        result["print"]["path"] = ctx.sandbox.relative(target / result["print"]["path"])
        result["epub"]["path"] = ctx.sandbox.relative(target / result["epub"]["path"])
        result["note"] = ("Nothing has looked at this painting. Curate it by hand before it goes anywhere near a "
                          "manuscript: open the files, keep or discard, and only then let the Leanpub preview "
                          "build (ADR 0001 of the Book Drafting Pipeline) be the gate that says it renders.")
        return result