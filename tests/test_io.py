"""Input/output contracts: what happens when only one path is connected,
what the placeholders look like, and that batches survive."""

import torch
from helpers import load_node, image, latent, run

mod = load_node()
XFAIL = set()


def test_image_only_gives_a_wellformed_latent_placeholder():
    r = run(mod, image=image(512, 512), method="lanczos", scale_factor=2.0)
    assert tuple(r[1]["samples"].shape) == (1, 4, 8, 8), r[1]["samples"].shape


def test_latent_only_gives_a_wellformed_image_placeholder():
    r = run(mod, samples=latent(64, 96), method="bicubic", scale_factor=2.0)
    assert tuple(r[0].shape) == (1, 8, 8, 3), r[0].shape
    assert tuple(r[1]["samples"].shape)[2:] == (192, 128), r[1]["samples"].shape


def test_neither_connected_is_a_clear_error():
    try:
        run(mod, method="lanczos")
    except ValueError as exc:
        assert "connect" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("expected a ValueError naming the missing input")


def test_latent_is_sized_from_its_own_dimensions():
    """Deliberately not image/8 -- the latent must not inherit the image's
    target, it must compute its own from the same spec."""
    r = run(mod, image=image(1024, 1536), samples=latent(100, 150),
            method="lanczos", target_mode="scale_factor", scale_factor=2.0,
            multiple_of="off")
    assert tuple(r[0].shape)[1:3] == (3072, 2048), r[0].shape
    assert tuple(r[1]["samples"].shape)[2:] == (300, 200), r[1]["samples"].shape


def test_batches_survive_both_paths():
    r = run(mod, image=image(256, 384, b=3), samples=latent(32, 48, b=2),
            method="lanczos", scale_factor=2.0)
    assert r[0].shape[0] == 3, r[0].shape
    assert r[1]["samples"].shape[0] == 2, r[1]["samples"].shape


def test_latent_dict_extras_are_carried_through():
    lat = latent(64, 64)
    lat["noise_mask"] = torch.ones(1, 64, 64)
    r = run(mod, samples=lat, method="bicubic", scale_factor=2.0)
    assert "noise_mask" in r[1], "extra latent keys must not be dropped"


def test_model_method_falls_back_when_nothing_is_selected():
    r = run(mod, image=image(256, 384), method="model", scale_factor=2.0)
    assert tuple(r[0].shape)[1:3] == (768, 512), r[0].shape
    assert "lanczos" in r[4], r[4]


def test_model_is_skipped_on_a_downscale():
    r = run(mod, image=image(1024, 1536), method="model", scale_factor=0.5,
            upscale_model="whatever.pth")
    assert tuple(r[0].shape)[1:3] == (768, 512), r[0].shape
    assert "downscale" in r[4], r[4]


def test_width_and_height_outputs_match_the_image():
    r = run(mod, image=image(1024, 1536), method="lanczos", scale_factor=2.0)
    assert (r[2], r[3]) == (int(r[0].shape[2]), int(r[0].shape[1]))


def test_all_kernels_run():
    for k in ["lanczos", "bicubic", "bilinear", "nearest", "area"]:
        r = run(mod, image=image(256, 384), samples=latent(32, 48),
                method=k, scale_factor=1.5)
        assert r[0].shape[1] > 0 and r[1]["samples"].shape[2] > 0, k


def test_none_values_from_the_frontend_do_not_crash():
    """ComfyUI can send an untouched optional widget as an explicit null."""
    r = run(mod, image=image(256, 256), method="lanczos",
            target_mode=None, multiple_of=None, scale_factor=None,
            crop=None, crop_ratio=None, crop_orientation=None,
            crop_width=None, crop_height=None,
            crop_position=None, crop_offset_x=None, crop_offset_y=None)
    assert r[0].shape[0] == 1
