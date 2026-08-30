# Upscale Crop Universal

**Upscale and crop in one ComfyUI node — with the model built in.**

Five ways to ask for a size, all of them aspect-preserving and all of them working
downwards as well as up. Then, optionally, a crop stage that cuts the exact format
out of the result. The upscale model is picked by a widget straight from your
`models/upscale_models` folder, so there is no Load Upscale Model node to wire up.

Everything runs in torch and handles batches.

## The two stages

```
             ┌─────────── UPSCALE ───────────┐  ┌──── CROP ─────┐
image ──────►│ percent · factor · shortest   │─►│ 4:5 1:1 3:2   │──► image
latent ─────►│ longest · megapixels          │  │ DIN 16:9 …    │──► latent
             │ aspect always preserved       │  │ off by default│──► width/height/info
             └───────────────────────────────┘  └───────────────┘
```

**Upscale** sets the size and never changes the shape. **Crop** takes the format
out of it, and only ever cuts — nothing is resampled a second time.

## Asking for a size

| `target_mode` | what it pins | when it is the natural one |
|---|---|---|
| `scale_percent` | 200% = double | you think in percent |
| `scale_factor` | ×2.0 = double | you think in multipliers |
| `shortest_side` | `min(w, h)` | feeding a model with a minimum edge |
| `longest_side` | `max(w, h)` | fitting a print or a screen |
| `megapixels` | total pixel count | staying inside a VRAM or upload budget |

Percent and factor are the same arithmetic, offered both ways because which one
feels natural depends on the task. Every mode downscales when you give it a value
below the current size — set `scale_factor` to 0.5 and this is your downscaler.

`multiple_of` snaps the result so it survives a VAE. **8** suits SD, SDXL and Flux;
`off` gives you the exact number you asked for, which is fine when the image is
going straight to disk.

## The crop stage

Off by default, so the node is a pure upscaler until you want a format.

| format | |
|---|---|
| `4:5` | social portrait, 8×10 print — **the default** |
| `1:1` | square |
| `4:3` | classic monitor, Four Thirds |
| `3:2` | 35 mm, most DSLRs |
| `DIN` | √2 — A4 and the whole A series |
| `16:10` | widescreen monitor |
| `16:9` | HD video |
| `2:1` | univisium |
| `21:9` | cinemascope |
| `custom` | your own `crop_width` × `crop_height` |

Each format is written the way people say it — 4:5 is portrait, 16:9 is landscape —
and `crop_orientation` flips whichever you pick, so there are no mirrored duplicates
in the list. Picking a format takes the **largest window of that shape that fits**.

`crop_position` (center / top / bottom / left / right) and `crop_offset` decide what
survives — the same controls as the JPS crop nodes. The offset is clamped to the
slack the anchor leaves, so the window can never wander off the image.

## The model

`method` defaults to **`model`**, and `upscale_model` lists your
`models/upscale_models` folder directly. One pass at the model's native scale, then
a single resize onto the target — so a 4× model asked for a 2× result does not leave
you with a needlessly huge intermediate. Tiling reuses ComfyUI's own `tiled_scale`
and halves the tile if VRAM runs short.

It gets out of the way when it should: a pure downscale skips the model entirely,
and a missing or unselected model falls back to lanczos. Either way the `info`
output says what actually happened rather than failing silently.

The other methods — `lanczos`, `bicubic`, `bilinear`, `nearest`, `area` — are plain
kernels. lanczos is the sharpest non-model choice; area is the best for heavy
downscaling.

## Image and latent

Both inputs are optional; connect either or both. Each is sized from **its own**
dimensions rather than assuming the latent is the image divided by eight, and the
crop is applied in normalised 0–1 coordinates so the two come out framed
identically. Whichever you did not connect comes back as a small well-formed
placeholder, so the output sockets are always safe to leave dangling.

Connecting neither is an error rather than a silent no-op — the node would have
nothing to do.

## Outputs

`image`, `latent`, `width`, `height`, `info` — where `info` is a one-line readout of
the sizes, the method actually used, the crop applied, and any fallback that kicked in.

## Interface

The widget layout was designed before any of it was written:
[`docs/interface.html`](docs/interface.html) is an interactive mockup of the node —
click the coloured rows to see which widgets appear for which mode.

Only the widgets that currently apply are shown. Switching `target_mode` swaps the
size field, `method` reveals or hides the model picker, and `crop` folds the whole
second block away.

## Install

Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. No dependencies beyond
ComfyUI itself (`spandrel`, used to load upscale models, already ships with it).

## Tests

```
python tests/run_tests.py
```

30 tests, no pytest needed — the portable ComfyUI python does not have it. They run
against a stubbed `folder_paths` and never touch a real model, so they work anywhere
torch does.
