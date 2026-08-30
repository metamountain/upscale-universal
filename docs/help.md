# Upscale Crop Universal — full reference

Every widget, what it does, and what it interacts with. The short version is in
the [README](../README.md); this is the page to reach for when something behaves
in a way you did not expect.

Every widget also carries this text as a **hover tooltip in the node itself**, so
you rarely need to come here.

---

## Finding the node

Double-click the canvas and type any of: **upscale**, **resize**, **rescale**,
**downscale**, **crop**, **aspect ratio**, **shortest side**, **longest side**,
**megapixels**, **resolution**, **esrgan**, **ultrasharp**, **16:9**, **4:5**,
**portrait**, **landscape**, **exact size**.

It lives under **image › upscaling**, next to ComfyUI's own upscale nodes.

---

## Why widgets disappear

**This is the single most common surprise.** Only the widgets that currently
apply are drawn:

| you change | what appears / vanishes |
|---|---|
| `target_mode` | the one matching size field; the other four hide |
| `method` → `model` | `upscale_model` appears |
| `method` → anything else | `upscale_model` hides |
| `crop` → on | the whole crop block appears |
| `crop_ratio` → `custom` | `crop_width` / `crop_height` replace `crop_orientation` |
| `crop_ratio` → `1:1` | `crop_orientation` hides — a square has no orientation |

A hidden widget **keeps its value**. It is not reset, it is just not in use.
Switch back and it is exactly as you left it. The `info` output always names the
mode and method that actually ran, so if you are unsure what took effect, read
that.

---

## Upscale

### `target_mode`

Five ways of asking for a size. All keep the aspect ratio; none of them can
distort the image. All work **downwards** too — a value below the current size
downscales, and that is a supported use, not a mistake.

| mode | pins | reach for it when |
|---|---|---|
| `scale_percent` | `200%` = double | you think in percent |
| `scale_factor` | `×2.0` = double | you think in multipliers |
| `shortest_side` | `min(w, h)` | a model needs a minimum edge |
| `longest_side` | `max(w, h)` | fitting a print, a screen, an upload limit |
| `megapixels` | total pixel count | staying inside a VRAM or filesize budget |

`scale_percent` and `scale_factor` are the same arithmetic — `200%` and `×2.0`
produce identical output. Both exist because which one feels natural depends on
what you are doing, and translating in your head is a small tax you should not
have to pay.

### `method`

| method | notes |
|---|---|
| `model` | **default.** Uses `upscale_model` below. See [The model](#the-model). |
| `lanczos` | sharpest non-model choice; the general-purpose default if you skip the model |
| `bicubic` | smoother than lanczos, fewer ringing artifacts on soft material |
| `bilinear` | soft; rarely what you want for an upscale |
| `nearest` | hard pixel edges — for pixel art, or when you must not invent values |
| `area` | **best for heavy downscaling**; averages instead of sampling, so it does not alias |

On the latent path `model` and `lanczos` both fall through to `bicubic`, because
an upscale model works on pixels and torch has no lanczos for tensors.

### `upscale_model`

Only visible when `method = model`. Lists the files in your
`models/upscale_models` folder directly — **there is no Load Upscale Model node
to wire up**.

If the folder is empty the widget says so, and the node falls back to lanczos
rather than failing.

### `multiple_of`

Snaps the final width and height to a multiple, so the result survives whatever
comes next.

| value | when |
|---|---|
| `8` | **default.** SD, SDXL and Flux all encode in 8-pixel blocks |
| `16`, `32`, `64` | tiling workflows and some video models |
| `2`, `4` | rarely needed |
| `off` | you get the exact number you asked for — fine when the image goes straight to disk |

Without this, a size like `2046×3070` reaches the VAE, which quietly crops it or
errors — and the cause is not obvious from the message you get. The `info` output
says when snapping changed your number.

On the latent path the snap is divided by 8, because a latent cell is eight
pixels. Snapping a latent to a multiple of 8 directly would round it eight times
too coarsely.

---

## Crop

Off by default. Switched on it **only ever cuts** — never rescales — so nothing is
resampled a second time.

### `crop_ratio`

| format | |
|---|---|
| `4:5` | **default.** Social portrait, 8×10 print |
| `1:1` | square |
| `4:3` | classic monitor, Four Thirds |
| `3:2` | 35 mm, most DSLRs |
| `DIN` | √2 — A4 and the whole A series |
| `16:10` | widescreen monitor |
| `16:9` | HD video |
| `2:1` | univisium |
| `21:9` | cinemascope |
| `custom` | your own `crop_width` × `crop_height` |

Picking a format takes the **largest window of that shape that fits** inside the
upscaled image. You never get a window bigger than the image, and you never get
padding.

### `crop_orientation`

Flips the chosen format. Hidden for `1:1` and for `custom`.

Each format above is written the way people say it — `4:5` is portrait, `16:9` is
landscape — and orientation normalises whichever you pick. That is why the list
has no mirrored duplicates like `5:4` or `9:16`: they are the same format with
this switch thrown.

### `crop_width` / `crop_height`

Only visible when `crop_ratio = custom`. Exact pixel dimensions. Still subject to
`multiple_of`, so asking for `900` with `multiple_of 8` gives you `896` — the
`info` output says so. Larger than the image is clamped to the image.

### `crop_position` and `crop_offset`

Where the window sits, and a nudge from there. The same controls as the JPS crop
nodes, so if you have used those this is familiar.

| position | window sits | offset moves it |
|---|---|---|
| `center` | centred both ways | up / down |
| `top` | flush to the top | down |
| `bottom` | flush to the bottom | up |
| `left` | flush to the left | right |
| `right` | flush to the right | left |

The offset is **clamped to the slack the anchor leaves**, so the window can never
wander off the image. Pushing past the edge simply stops there rather than giving
you a short crop or an error.

---

## Inputs and outputs

### `image` and `samples` — both optional

Connect either or both.

Each is sized from **its own dimensions**. A latent is not assumed to be the image
divided by eight — connect a latent whose size has no relation to the image and
both still land on their own correct target.

The crop is computed in **normalised 0–1 coordinates**, so an image and a latent
come out framed identically even though their pixel dimensions differ.

Whichever you did not connect comes back as a small well-formed placeholder
(`1×8×8×3` for image, `1×4×8×8` for latent), so the output sockets are always safe
to leave dangling.

Connecting **neither** raises an error rather than silently doing nothing — the
node would have no work to do, and a graph that appears to run while achieving
nothing is worse than one that stops and says why.

### Outputs

| output | |
|---|---|
| `image` | the result, or the placeholder |
| `latent` | the result, or the placeholder. Extra keys in the latent dict (`noise_mask` and friends) are carried through |
| `width` / `height` | the final dimensions as integers, for feeding a downstream node |
| `info` | a one-line readout of what actually happened |

Read `info` when something surprises you. It names the sizes in and out, the mode,
the method that actually ran, the crop applied, and any fallback that kicked in.

---

## The model

`method = model` runs **one pass** at the model's native scale, then a single
resize onto your target.

One pass, not several, and this is deliberate: running a 4× model twice gives 16×,
far past anything useful, and the artifacts compound. If your target needs more
than the model's native scale, the remainder is covered by lanczos — softer than a
second model pass, but without the artifact stacking.

A 4× model asked for a 2× target upscales once to 4× and then comes back down to
2×. That is the right order — the model sees the original pixels, and the
reduction afterwards is cheap and clean.

**It gets out of the way when it should:**

| situation | what happens | `info` says |
|---|---|---|
| target is smaller than the source | model skipped entirely | `model skipped, target is a downscale` |
| no model selected | falls back to lanczos | `no upscale model selected` |
| model fails to load | falls back to lanczos | `model failed (…)` |

Tiling reuses ComfyUI's own `tiled_scale` and halves the tile if VRAM runs short,
down to 128px before giving up. There is nothing to configure — no tile size, no
pass count. On a card with real VRAM you will never see it engage.

The loaded model is cached, so dragging a slider does not reload it from disk.

---

## The live preview

The node shows what the crop will cut, live, without queueing the graph. Original
image, discarded area dimmed, the surviving window framed with its output size on
it.

It needs **one full run first** — the image only exists inside the node while it
executes, so before that there is nothing to draw and the panel says so.

Two things it deliberately does not do:

- **It does not re-render the upscale.** An upscale looks identical at thumbnail
  size, so re-running it per keystroke would cost real time to show nothing new.
  What changes with the sliders are the numbers and the crop rectangle.
- **It does not guess.** The sizes and the rectangle come from the same functions
  the node itself runs, so what the preview reports is what the graph produces.
  This is pinned by tests.

---

## Tests

```
python tests/run_tests.py
```

37 tests, no pytest needed — the portable ComfyUI python does not ship it. They
run against a stubbed `folder_paths` and never touch a real model, so they work
anywhere torch does.

**What they do not cover:** the `method = model` path with real weights. Only its
fallbacks are tested (downscale skip, missing model, load failure). The tiling and
the model call itself need a real ComfyUI to exercise.
