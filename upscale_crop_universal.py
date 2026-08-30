"""
Upscale Crop Universal — ComfyUI custom node

Two stages in one node:

  UPSCALE  sets the size and always keeps the aspect ratio. Percent, factor,
           shortest side, longest side or megapixels -- five ways of asking
           for the same thing, because which one is natural depends on why
           you are resizing. Every one of them works in both directions; a
           value below the current size downscales.

  CROP     takes the exact format out of the result. Off by default, so the
           node is a pure upscaler until you want a format. Switched on it
           only ever cuts, never rescales, so nothing is resampled twice.

The upscale model is picked by a widget, not wired in from a loader node,
and `method = model` is the default -- one pass at the model's native scale,
then a single resize onto the target.

Image and latent are both optional and both handled. Each is sized from its
own dimensions rather than assuming the latent is the image divided by eight,
and the crop is expressed in normalised 0..1 coordinates so both come out
geometrically identical.

All torch -> runs on the GPU and handles whole batches.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths

# ComfyUI's own helpers. Guarded so the module still imports outside a running
# ComfyUI -- tests/run_tests.py has no ComfyUI to import.
try:
    import comfy.model_management as _mm
    import comfy.utils as _cu
except Exception:
    _mm = None
    _cu = None


def _device():
    if _mm is not None:
        return _mm.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- sizing

TARGET_MODES = ["scale_percent", "scale_factor", "shortest_side",
                "longest_side", "megapixels"]

MULTIPLES = ["off", "2", "4", "8", "16", "32", "64"]

# Interpolation vocabulary follows ComfyUI core and the JPS crop nodes, so a
# name means the same thing here as it does everywhere else in a graph.
METHODS = ["model", "lanczos", "bicubic", "bilinear", "nearest", "area"]

# Classic formats, each written the way people actually say it -- 4:5 is
# portrait, 16:9 is landscape -- and crop_orientation normalises whichever
# one you pick, so the table needs no mirrored duplicates.
RATIOS = {
    "4:5":   (4.0, 5.0),
    "1:1":   (1.0, 1.0),
    "4:3":   (4.0, 3.0),
    "3:2":   (3.0, 2.0),
    "DIN":   (2.0 ** 0.5, 1.0),
    "16:10": (16.0, 10.0),
    "16:9":  (16.0, 9.0),
    "2:1":   (2.0, 1.0),
    "21:9":  (64.0, 27.0),
}
RATIO_NAMES = ["custom"] + list(RATIOS.keys())

CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]
ORIENTATIONS = ["portrait", "landscape"]


def _snap(n, mult):
    """Round to the nearest multiple, never below one multiple."""
    n = max(1.0, float(n))
    if mult <= 1:
        return max(1, int(round(n)))
    return max(mult, int(round(n / mult)) * mult)


def _scale_for(w, h, mode, percent, factor, shortest, longest, megapixels):
    """The factor a mode asks for, given a tensor's own size.

    Every mode returns a plain multiplier, so the caller applies it the same
    way regardless of which one the user picked -- and a multiplier below 1.0
    is a downscale, which is deliberately not clamped away.
    """
    if mode == "scale_percent":
        return max(0.01, percent / 100.0)
    if mode == "scale_factor":
        return max(0.01, factor)
    if mode == "shortest_side":
        return shortest / max(1, min(w, h))
    if mode == "longest_side":
        return longest / max(1, max(w, h))
    if mode == "megapixels":
        return (megapixels * 1e6 / max(1, w * h)) ** 0.5
    return 1.0


def _target(w, h, k, mult):
    return _snap(w * k, mult), _snap(h * k, mult)


# ---------------------------------------------------------------- crop rect

def _crop_rect(w, h, ratio, orientation, cw, ch, position, offset):
    """The surviving region as normalised 0..1 (x0, y0, x1, y1).

    Normalised rather than pixels so the identical geometry can be applied to
    an image and to a latent of a completely different size -- the two then
    come out framed the same, which they would not if the latent were cropped
    with pixel numbers meant for the image.
    """
    if ratio == "custom":
        box_w = min(float(cw), float(w))
        box_h = min(float(ch), float(h))
    else:
        rw, rh = RATIOS[ratio]
        ar = rw / rh
        if ratio != "1:1":
            # normalise to the requested orientation whichever way the
            # table happens to write this format
            want_wide = orientation == "landscape"
            if want_wide != (ar >= 1.0):
                ar = 1.0 / ar
        # largest window of that shape that fits inside
        if (w / max(1.0, float(h))) > ar:
            box_h = float(h)
            box_w = box_h * ar
        else:
            box_w = float(w)
            box_h = box_w / ar

    box_w = max(1.0, min(box_w, float(w)))
    box_h = max(1.0, min(box_h, float(h)))

    slack_x = float(w) - box_w
    slack_y = float(h) - box_h

    # Anchor first, then nudge -- and only ever within the slack the anchor
    # actually leaves, so the window can never wander off the image.
    if position == "left":
        x0 = min(max(0.0, offset), slack_x)
        y0 = slack_y / 2.0
    elif position == "right":
        x0 = slack_x + min(0.0, max(-slack_x, offset))
        y0 = slack_y / 2.0
    elif position == "top":
        x0 = slack_x / 2.0
        y0 = min(max(0.0, offset), slack_y)
    elif position == "bottom":
        x0 = slack_x / 2.0
        y0 = slack_y + min(0.0, max(-slack_y, offset))
    else:  # center
        x0 = slack_x / 2.0
        y0 = slack_y / 2.0 + max(-slack_y / 2.0, min(slack_y / 2.0, offset))

    return (x0 / w, y0 / h, (x0 + box_w) / w, (y0 + box_h) / h)


def _apply_rect(t, rect, mult, channels_last):
    """Cut a normalised rect out of a B,H,W,C or B,C,H,W tensor."""
    h = t.shape[1] if channels_last else t.shape[2]
    w = t.shape[2] if channels_last else t.shape[3]

    x0 = int(round(rect[0] * w))
    y0 = int(round(rect[1] * h))
    cw = _snap((rect[2] - rect[0]) * w, mult)
    ch = _snap((rect[3] - rect[1]) * h, mult)

    # snapping can push the window past the edge -- pull it back in rather
    # than returning a short crop
    cw = min(cw, w)
    ch = min(ch, h)
    x0 = max(0, min(x0, w - cw))
    y0 = max(0, min(y0, h - ch))

    if channels_last:
        return t[:, y0:y0 + ch, x0:x0 + cw, :]
    return t[:, :, y0:y0 + ch, x0:x0 + cw]


# ---------------------------------------------------------------- resampling

# torch has no lanczos, so the image path detours through PIL for that one
# kernel. Everything else stays in torch on the GPU.
_TORCH_MODES = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "nearest": "nearest-exact",
    "area": "area",
}


def _resize_image(img, tw, th, kernel):
    """B,H,W,C float 0..1 -> resized the same way."""
    if img.shape[1] == th and img.shape[2] == tw:
        return img

    if kernel == "lanczos":
        out = []
        for i in range(img.shape[0]):
            a = (img[i].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            p = Image.fromarray(a).resize((tw, th), Image.LANCZOS)
            out.append(torch.from_numpy(np.asarray(p).astype(np.float32) / 255.0))
        return torch.stack(out, 0).to(img.device)

    mode = _TORCH_MODES.get(kernel, "bicubic")
    x = img.movedim(-1, 1)
    kw = {} if mode in ("nearest-exact", "area") else {"align_corners": False}
    x = F.interpolate(x, size=(th, tw), mode=mode, **kw)
    return x.movedim(1, -1).clamp(0, 1)


def _resize_latent(lat, tw, th, kernel):
    """B,C,H,W -- no lanczos and no clamping; a latent is not 0..1 pixels."""
    if lat.shape[2] == th and lat.shape[3] == tw:
        return lat
    mode = _TORCH_MODES.get(kernel, "bicubic")   # 'model' lands here too
    kw = {} if mode in ("nearest-exact", "area") else {"align_corners": False}
    return F.interpolate(lat, size=(th, tw), mode=mode, **kw)


# ---------------------------------------------------------------- the model

_MODEL_CACHE = {"name": None, "model": None}


def _load_upscale_model(name):
    """Load and keep one model, so dragging a slider does not reload it."""
    if _MODEL_CACHE["name"] == name and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"]

    from spandrel import ModelLoader, ImageModelDescriptor

    path = folder_paths.get_full_path_or_raise("upscale_models", name)
    sd = _cu.load_torch_file(path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = _cu.state_dict_prefix_replace(sd, {"module.": ""})
    model = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(f"{name} is not a single-image upscale model.")

    _MODEL_CACHE["name"] = name
    _MODEL_CACHE["model"] = model
    return model


def _model_pass(img, name):
    """One pass at the model's native scale, tiled, shrinking tiles on OOM.

    This mirrors ComfyUI's own ImageUpscaleWithModel rather than reinventing
    the tiling, so a model that works in the core node works here too and the
    OOM retry behaves the way people already expect.
    """
    model = _load_upscale_model(name)
    device = _device()

    if _mm is not None:
        need = _mm.module_size(model.model)
        need += (512 * 512 * 3) * img.element_size() * max(model.scale, 1.0) * 384.0
        need += img.nelement() * img.element_size()
        _mm.free_memory(need, device)

    model.to(device)
    x = img.movedim(-1, -3).to(device)

    tile, overlap = 512, 32
    oom_exc = getattr(_mm, "OOM_EXCEPTION", RuntimeError) if _mm else RuntimeError
    while True:
        try:
            steps = x.shape[0] * _cu.get_tiled_scale_steps(
                x.shape[3], x.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = _cu.ProgressBar(steps)
            out = _cu.tiled_scale(x, lambda a: model(a), tile_x=tile, tile_y=tile,
                                  overlap=overlap, upscale_amount=model.scale, pbar=pbar)
            break
        except oom_exc:
            tile //= 2
            if tile < 128:
                model.to("cpu")
                raise

    model.to("cpu")
    return out.movedim(-3, -1).clamp(0, 1), float(model.scale)


# ---------------------------------------------------------------- node

class UpscaleCropUniversal:

    DESCRIPTION = (
        "Upscale and crop in one node. UPSCALE sets the size and always keeps the "
        "aspect ratio -- ask for it as a percent, a factor, a shortest or longest "
        "side, or a megapixel count, whichever is natural for what you are doing. "
        "Every mode works downwards too, so this is also your downscaler.\n\n"
        "CROP is off by default and only ever cuts, never rescales, so nothing is "
        "resampled twice. Switch it on and pick a classic format -- 4:5, 1:1, 3:2, "
        "DIN, 16:9 and the rest -- with crop_orientation flipping it and "
        "crop_position plus crop_offset deciding what survives.\n\n"
        "THE MODEL IS BUILT IN. method defaults to 'model' and upscale_model picks "
        "the file straight from your models/upscale_models folder -- no separate "
        "loader node to wire up. One pass at the model's native scale, then a single "
        "resize onto the target, so a 4x model hitting a 2x target does not leave you "
        "with a needlessly huge intermediate. Tiling is automatic and halves itself if "
        "VRAM runs short.\n\n"
        "IMAGE AND LATENT are both optional and both handled. Each is sized from its "
        "own dimensions rather than assuming the latent is the image divided by eight, "
        "and the crop is applied in normalised coordinates so the two come out framed "
        "identically. Connect either or both.\n\n"
        "multiple_of snaps the result so it survives a VAE -- 8 suits SD, SDXL and "
        "Flux; set it to off only when the image is going straight to disk."
    )

    SEARCH_ALIASES = [
        "upscale", "upscale image", "upscale model", "esrgan", "resize", "rescale",
        "scale image", "downscale", "crop", "crop image", "aspect ratio",
        "shortest side", "longest side", "megapixels", "resolution",
        "image scale", "upscale and crop", "format", "portrait", "landscape",
        # German
        "hochskalieren", "skalieren", "vergroessern", "verkleinern",
        "zuschneiden", "beschneiden", "seitenverhaeltnis", "bildformat",
        "aufloesung", "hochformat", "querformat",
    ]

    OUTPUT_TOOLTIPS = (
        "The resized (and optionally cropped) image. A small placeholder if no image was connected.",
        "The resized latent, framed to match the image. A small placeholder if no latent was connected.",
        "Final width in px.",
        "Final height in px.",
        "What actually happened: sizes, the method used, the crop applied, and any fallback.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("upscale_models")
        if not models:
            models = ["(no models in models/upscale_models)"]
        return {
            "required": {
                "target_mode": (TARGET_MODES, {"default": "scale_factor",
                                               "tooltip": "How you want to ask for the size. All five keep the aspect ratio and all five work downwards as well as up."}),
                "method": (METHODS, {"default": "model",
                                     "tooltip": "How to resample. 'model' uses the upscale_model below; the rest are plain kernels. lanczos is the sharpest non-model choice, area the best for heavy downscaling."}),
                "multiple_of": (MULTIPLES, {"default": "8",
                                            "tooltip": "Snap the result to this multiple so it survives a VAE. 8 suits SD, SDXL and Flux. 'off' gives you the exact number you asked for -- fine when the image goes straight to disk."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional. Connect either this, or samples, or both."}),
                "samples": ("LATENT", {"tooltip": "Optional. Sized from its own dimensions, never from the image's."}),
                "upscale_model": (models, {"tooltip": "Used when method is 'model'. Read straight from models/upscale_models -- no loader node needed."}),
                "scale_percent": ("FLOAT", {"default": 200.0, "min": 1.0, "max": 800.0, "step": 5.0,
                                            "tooltip": "Used by target_mode scale_percent. 200 doubles, 50 halves."}),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 8.0, "step": 0.05,
                                           "tooltip": "Used by target_mode scale_factor. The same thing as percent, expressed the other way."}),
                "shortest_side": ("INT", {"default": 1536, "min": 16, "max": 16384, "step": 8,
                                          "tooltip": "Used by target_mode shortest_side. Pins min(width, height); the other side follows the aspect ratio."}),
                "longest_side": ("INT", {"default": 2048, "min": 16, "max": 16384, "step": 8,
                                         "tooltip": "Used by target_mode longest_side. Pins max(width, height)."}),
                "megapixels": ("FLOAT", {"default": 2.0, "min": 0.05, "max": 64.0, "step": 0.05,
                                         "tooltip": "Used by target_mode megapixels. Pins the total pixel count; the shape is unchanged."}),
                "crop": ("BOOLEAN", {"default": False,
                                     "tooltip": "Off = pure upscaler. On = cut the format out of the result afterwards."}),
                "crop_ratio": (RATIO_NAMES, {"default": "4:5",
                                             "tooltip": "The format to cut. Takes the largest window of that shape that fits. 'custom' uses crop_width and crop_height instead."}),
                "crop_orientation": (ORIENTATIONS, {"default": "portrait",
                                                    "tooltip": "Flips the chosen format. Ignored for 1:1 and for custom."}),
                "crop_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8,
                                       "tooltip": "Only used when crop_ratio is 'custom'."}),
                "crop_height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8,
                                        "tooltip": "Only used when crop_ratio is 'custom'."}),
                "crop_position": (CROP_POSITIONS, {"default": "center",
                                                   "tooltip": "Where the window sits before crop_offset nudges it."}),
                "crop_offset": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 8,
                                        "tooltip": "Nudge the window in px, along whichever axis the anchor leaves free. Clamped so it can never leave the image."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "latent", "width", "height", "info")
    FUNCTION = "run"
    CATEGORY = "image/upscaling"

    def run(self, target_mode="scale_factor", method="model", multiple_of="8",
            image=None, samples=None, upscale_model=None,
            scale_percent=200.0, scale_factor=2.0, shortest_side=1536,
            longest_side=2048, megapixels=2.0,
            crop=False, crop_ratio="4:5", crop_orientation="portrait",
            crop_width=1024, crop_height=1024,
            crop_position="center", crop_offset=0, unique_id=None):

        # ComfyUI's frontend can send an untouched optional widget as an
        # explicit None instead of omitting it, which otherwise blows up in
        # the float()/int() conversion before this ever runs.
        if target_mode is None: target_mode = "scale_factor"
        if method is None: method = "model"
        if multiple_of is None: multiple_of = "8"
        if scale_percent is None: scale_percent = 200.0
        if scale_factor is None: scale_factor = 2.0
        if shortest_side is None: shortest_side = 1536
        if longest_side is None: longest_side = 2048
        if megapixels is None: megapixels = 2.0
        if crop is None: crop = False
        if crop_ratio is None: crop_ratio = "4:5"
        if crop_orientation is None: crop_orientation = "portrait"
        if crop_width is None: crop_width = 1024
        if crop_height is None: crop_height = 1024
        if crop_position is None: crop_position = "center"
        if crop_offset is None: crop_offset = 0

        if image is None and samples is None:
            raise ValueError(
                "Upscale Crop Universal: connect an image, a latent, or both — "
                "there is nothing to resize.")

        mult = 1 if multiple_of == "off" else int(multiple_of)
        notes = []

        # ---- image path ------------------------------------------------
        out_image = None
        img_dims = None
        if image is not None:
            src_h, src_w = int(image.shape[1]), int(image.shape[2])
            k = _scale_for(src_w, src_h, target_mode, scale_percent, scale_factor,
                           shortest_side, longest_side, megapixels)
            tw, th = _target(src_w, src_h, k, mult)

            kernel = method
            work = image
            if method == "model":
                if k <= 1.0:
                    kernel = "lanczos"
                    notes.append("model skipped, target is a downscale")
                elif _cu is None:
                    kernel = "lanczos"
                    notes.append("comfy.utils unavailable, fell back to lanczos")
                elif not upscale_model or upscale_model.startswith("(no models"):
                    kernel = "lanczos"
                    notes.append("no upscale model selected, fell back to lanczos")
                else:
                    try:
                        work, native = _model_pass(image, upscale_model)
                        kernel = "lanczos"      # the trim onto the exact target
                        notes.append(f"{upscale_model} x{native:g}")
                    except Exception as exc:
                        work = image
                        kernel = "lanczos"
                        notes.append(f"model failed ({type(exc).__name__}), fell back to lanczos")

            out_image = _resize_image(work, tw, th, kernel)
            img_dims = (tw, th)

        # ---- latent path (never model; an upscale model is pixel-space) --
        out_latent = None
        if samples is not None:
            lat = samples["samples"]
            lh, lw = int(lat.shape[2]), int(lat.shape[3])
            k = _scale_for(lw, lh, target_mode, scale_percent, scale_factor,
                           shortest_side, longest_side, megapixels)
            # a latent's own multiple_of is in latent cells, not pixels
            lmult = max(1, mult // 8) if mult > 1 else 1
            ltw, lth = _target(lw, lh, k, lmult)
            out_latent = _resize_latent(lat, ltw, lth, method)

        # ---- crop, in normalised coordinates so both match ----------------
        crop_note = "crop off"
        if crop:
            ref_w, ref_h = img_dims if img_dims else (out_latent.shape[3], out_latent.shape[2])
            rect = _crop_rect(ref_w, ref_h, crop_ratio, crop_orientation,
                              crop_width, crop_height, crop_position, crop_offset)
            if out_image is not None:
                out_image = _apply_rect(out_image, rect, mult, channels_last=True)
                img_dims = (int(out_image.shape[2]), int(out_image.shape[1]))
            if out_latent is not None:
                lmult = max(1, mult // 8) if mult > 1 else 1
                out_latent = _apply_rect(out_latent, rect, lmult, channels_last=False)
            shape = crop_ratio if crop_ratio == "custom" else (
                crop_ratio + ("" if crop_ratio == "1:1" else " " + crop_orientation))
            crop_note = f"crop {shape} {crop_position}"
            if crop_offset:
                crop_note += f"{crop_offset:+d}"

        # ---- outputs, well-typed either way ------------------------------
        if out_image is None:
            out_image = torch.zeros((1, 8, 8, 3))
        if out_latent is None:
            out_latent = torch.zeros((1, 4, 8, 8))

        final = samples.copy() if samples is not None else {}
        final["samples"] = out_latent

        w = int(out_image.shape[2]) if image is not None else int(out_latent.shape[3])
        h = int(out_image.shape[1]) if image is not None else int(out_latent.shape[2])

        parts = []
        if image is not None:
            parts.append(f"image {image.shape[2]}x{image.shape[1]} -> {out_image.shape[2]}x{out_image.shape[1]}")
        if samples is not None:
            parts.append(f"latent {samples['samples'].shape[3]}x{samples['samples'].shape[2]}"
                         f" -> {out_latent.shape[3]}x{out_latent.shape[2]}")
        parts.append(target_mode)
        parts.append(method)
        parts.append(f"/{multiple_of}")
        parts.append(crop_note)
        parts.extend(notes)

        return (out_image, final, w, h, " | ".join(parts))
