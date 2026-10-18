# Painting book art (optional)

Jig is a text LLM: it doesn't paint. With the optional painter, Jig gets a *tool* - an image-model call on
AWS Bedrock - and a *skill* that knows how to prompt it. Jig orchestrates; the image model paints. The
painter is off by default: unless you turn it on, the `paint_image` tool is simply not offered to the model
and nothing about Jig's usual text work changes.

## Why a Rembrandt house style

The book-art pipeline draws chapter and cover images for the Leanpub books. To keep a whole book's art
visually coherent, the style is locked by **image-conditioned references**, not by prompt words: each
generation is conditioned on a fixed set of public-domain Rembrandt paintings (The Night Watch, The Syndics
of the Drapers' Guild, a self-portrait). Rembrandt died in 1669, so every one is public domain - the style
never comes from a copyrighted source. The modern-tech *subject* is whatever the prompt names: a server
room, a cloud, an MCP handshake, a computer by candlelight.

The painter produces **two derivatives from one generation**:

- `{slug}_print.png` - the full-resolution painting written with 300 DPI metadata, so Leanpub's print PDF
  sizes it without resampling.
- `{slug}_epub.jpg` - the same pixels scaled to a reader-safe width and compressed for EPUB/MOBI.

Both land under `manuscript/images/` in the book repo - the same tree the existing rendered diagrams use -
and reference the Leanpub preview-build gate (ADR 0001 of the Book Drafting Pipeline) as the check that
decides whether an image is usable.

## Vision stays off

There is **no self-review loop**. The painter generates blind and returns what the model painted without
judging it - a human curates every image before it goes anywhere near a manuscript. The tool's result says
exactly that, and no quality heuristic pretends to see what it can't.

## Turning it on

```toml
[painter]
enabled = true
# The default Stability AI SD3.5 Large on Bedrock; Amazon Titan Image Generator v2 is the alternative.
model_id = "stability.sd3-5-large-v1:0"
region = "us-east-1"
access_key_id_env = "AWS_ACCESS_KEY_ID"
access_key_secret_env = "AWS_SECRET_ACCESS_KEY"
# Reference images shipped with Jig under assets/painter/rembrandt; point here if you keep your own copy.
# reference_dir = "/path/to/rembrandt"
```

The AWS keys are read from the named environment variables (or the standard AWS ones); nothing is written to
the vault or disk. The AWS account must have model access for the region.

## Promoting the skill

The painter skill ships as a versioned definition at `skills/rembrandt-tech-painter.json`. Load it into the
skill bank exactly like the agent-authored ones:

```
jig skills import skills/rembrandt-tech-painter.json
jig skills import --activate skills/rembrandt-tech-painter.json
```

Without `--activate` it is a candidate that changes nothing; the `--activate` flag is the human's own
decision to promote it, and once active it appears in the index of approved skills like any other. Register
it if you have your own config file: `jig --config /path/to/jig.toml skills import ...`.

## A real generation needs a human and AWS

The unit tests cover the tool-call shape and the two-derivative contract offline with a fake driver - no AWS
credentials are created or reached. A real painting needs: the AWS keys exported, `paint_image` called for
one chapter's subject, the output opened and curated by hand, and then a real Leanpub preview build
(`POST /{slug}/preview.json` → poll `job_status.json` → check `pdf_preview_url`/`epub_preview_url`) before
the image counts as done.