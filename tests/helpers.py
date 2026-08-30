"""Shared bits for the standalone test runner."""

import importlib.util
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_node():
    """Import upscale_crop_universal.py without a running ComfyUI.

    The module imports folder_paths, which only exists inside ComfyUI, so a
    stub stands in for it. Everything the tests touch is pure tensor maths
    that does not need the real thing.
    """
    if "folder_paths" not in sys.modules:
        stub = types.ModuleType("folder_paths")
        stub.get_filename_list = lambda kind: []
        stub.get_full_path_or_raise = lambda kind, name: name
        stub.get_temp_directory = lambda: "/tmp"
        sys.modules["folder_paths"] = stub

    path = os.path.join(ROOT, "upscale_crop_universal.py")
    spec = importlib.util.spec_from_file_location("ucu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def image(w, h, b=1):
    """A B,H,W,C image with structure, so a resize has something to lose."""
    ys = torch.linspace(0, 1, h).view(1, h, 1, 1)
    xs = torch.linspace(0, 1, w).view(1, 1, w, 1)
    base = (ys * xs).repeat(b, 1, 1, 3)
    return (base + torch.rand(b, h, w, 3) * 0.1).clamp(0, 1)


def latent(w, h, b=1, c=4):
    return {"samples": torch.randn(b, c, h, w)}


def run(mod, **kw):
    return mod.UpscaleCropUniversal().run(**kw)
