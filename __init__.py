"""Upscale Crop Universal — ComfyUI custom node package."""

from .upscale_crop_universal import UpscaleCropUniversal

# ComfyUI's node search matches against the description, and there is no
# separate keywords field, so the terms people actually type are folded in
# here -- kept out of the prose itself so the hover tooltip stays readable.
UpscaleCropUniversal.DESCRIPTION += (
    "\n\nAlso found as: " + UpscaleCropUniversal.SEARCH_TERMS + "."
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"UpscaleCropUniversal": UpscaleCropUniversal}
NODE_DISPLAY_NAME_MAPPINGS = {"UpscaleCropUniversal": "Upscale Crop Universal"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
