"""The crop stage: formats, anchors, offset clamping, and the promise that
an image and a latent come out framed identically."""

from helpers import load_node, image, latent, run

mod = load_node()
XFAIL = set()

SRC_W, SRC_H = 1024, 1536


def _crop(**kw):
    kw.setdefault("image", image(SRC_W, SRC_H))
    kw.setdefault("method", "lanczos")
    kw.setdefault("target_mode", "scale_factor")
    kw.setdefault("scale_factor", 2.0)
    kw.setdefault("crop", True)
    r = run(mod, **kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[4]


def test_crop_off_leaves_the_upscale_alone():
    r = run(mod, image=image(SRC_W, SRC_H), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0, crop=False)
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (2048, 3072)
    assert "crop off" in r[4]


def test_default_45_portrait_on_a_portrait_source():
    w, h, _ = _crop(crop_ratio="4:5", crop_orientation="portrait")
    assert (w, h) == (2048, 2560), (w, h)
    assert abs(w / h - 4 / 5) < 0.01


def test_orientation_flips_the_format():
    p = _crop(crop_ratio="4:5", crop_orientation="portrait")[:2]
    l = _crop(crop_ratio="4:5", crop_orientation="landscape")[:2]
    assert p[0] / p[1] < 1 < l[0] / l[1], (p, l)


def test_every_format_fits_inside_and_keeps_its_shape():
    """Whichever way the table writes a format, orientation normalises it:
    landscape is always the wide reading, portrait always the tall one."""
    want = {"4:5": 4 / 5, "1:1": 1.0, "4:3": 4 / 3, "3:2": 3 / 2,
            "DIN": 2 ** 0.5, "16:10": 1.6, "16:9": 16 / 9, "2:1": 2.0,
            "21:9": 64 / 27}
    for name, raw in want.items():
        wide, tall = max(raw, 1 / raw), min(raw, 1 / raw)
        for orient, ar in (("landscape", wide), ("portrait", tall)):
            w, h, _ = _crop(crop_ratio=name, crop_orientation=orient,
                            multiple_of=1)
            assert w <= 2048 and h <= 3072, (name, orient, w, h)
            assert abs(w / h - ar) < 0.02, (name, orient, w / h, ar)


def test_custom_uses_the_given_pixels():
    w, h, _ = _crop(crop_ratio="custom", target_width=1200, target_height=900,
                    multiple_of=1)
    assert (w, h) == (1200, 900), (w, h)


def test_custom_pixels_still_obey_multiple_of():
    """900 is not a multiple of 8, so it must come back snapped, not exact."""
    w, h, _ = _crop(crop_ratio="custom", target_width=1200, target_height=900,
                    multiple_of=8)
    assert (w, h) == (1200, 896), (w, h)


def test_custom_larger_than_the_image_is_clamped():
    w, h, _ = _crop(crop_ratio="custom", target_width=9000, target_height=9000)
    assert w <= 2048 and h <= 3072, (w, h)


def test_offset_cannot_push_the_window_off_the_image():
    """A silly offset must still yield the same size, just shifted as far
    as the image allows -- never a short crop or a crash."""
    base = _crop(crop_ratio="16:9", crop_orientation="landscape")[:2]
    for off in (-8192, -500, 0, 500, 8192):
        for pos in ("center", "top", "bottom", "left", "right"):
            w, h, _ = _crop(crop_ratio="16:9", crop_orientation="landscape",
                            crop_position=pos, crop_offset_y=off)
            assert (w, h) == base, (pos, off, w, h, base)


def test_all_anchors_run_and_keep_the_size():
    base = None
    for pos in ("center", "top", "bottom", "left", "right"):
        w, h, _ = _crop(crop_ratio="1:1", crop_position=pos)
        if base is None:
            base = (w, h)
        assert (w, h) == base, (pos, w, h, base)


def test_crop_output_respects_multiple_of():
    for m in [8, 16, 64]:
        w, h, _ = _crop(crop_ratio="DIN", crop_orientation="portrait", multiple_of=int(m))
        assert w % int(m) == 0 and h % int(m) == 0, (m, w, h)


def test_image_and_latent_are_framed_the_same():
    """The whole reason the crop is computed in normalised coordinates."""
    r = run(mod, image=image(SRC_W, SRC_H), samples=latent(128, 192),
            method="lanczos", target_mode="scale_factor", scale_factor=2.0,
            crop=True, crop_ratio="16:9", crop_orientation="landscape",
            multiple_of=1)
    iw, ih = int(r[0].shape[2]), int(r[0].shape[1])
    lw, lh = int(r[1]["samples"].shape[3]), int(r[1]["samples"].shape[2])
    assert abs(iw / ih - lw / lh) < 0.05, (iw, ih, lw, lh)
