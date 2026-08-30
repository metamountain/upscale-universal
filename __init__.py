"""Upscale Crop Universal — ComfyUI custom node package."""

from .upscale_crop_universal import UpscaleCropUniversal

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {"UpscaleCropUniversal": UpscaleCropUniversal}
NODE_DISPLAY_NAME_MAPPINGS = {"UpscaleCropUniversal": "Upscale Crop Universal"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
