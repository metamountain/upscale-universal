"""target_mode = custom, the box mode, and the crop controls around it.

It is the one mode that cannot preserve the aspect ratio: it scales until the
target_width x target_height box is covered, then crops to it, so the
resolution you type is the resolution you get. The crop is not optional here.

That is NOT the same as crop_ratio = custom, which never rescales -- it only
cuts the largest window of that shape out of whatever is already there.
"""

from helpers import load_node, image, latent, run

mod = load_node()
XFAIL = set()


def _box(w, h, src=(1024, 1536), **kw):
    kw.setdefault("method", "lanczos")
    r = run(mod, image=image(*src), target_mode="custom",
            target_width=w, target_height=h, **kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[4]


def test_you_get_exactly_the_box():
    for w, h in [(1920, 1080), (1080, 1920), (1000, 1000), (2048, 512)]:
        got = _box(w, h)[:2]
        assert got == (w, h), ((w, h), got)


def test_the_box_is_hit_from_any_source_shape():
    for src in [(1024, 1536), (1536, 1024), (512, 512), (3000, 400)]:
        got = _box(1920, 1080, src=src)[:2]
        assert got == (1920, 1080), (src, got)


def test_it_covers_rather_than_fits():
    """Covering means no bars. A fit would have left one axis short, and this
    node never pads, so every pixel of the box must come from the image."""
    w, h, _ = _box(1920, 1080, src=(1024, 1536))
    assert (w, h) == (1920, 1080)
    # 1024x1536 is portrait; covering a landscape box scales by width
    # (1920/1024 = 1.875) giving 1920x2880, then crops the height down.


def test_it_downscales_when_the_box_is_smaller():
    got = _box(256, 256, src=(2048, 2048))[:2]
    assert got == (256, 256), got


def test_the_box_obeys_multiple_of():
    """900 sits exactly halfway between 896 and 904, and Python rounds a tie
    to even -- 112.5 -> 112 -> 896. Worth pinning so a future refactor to
    round-half-up is noticed rather than silently shifting everyone's crops."""
    w, h, _ = _box(1000, 900, multiple_of=8)
    assert w % 8 == 0 and h % 8 == 0, (w, h)
    assert (w, h) == (1000, 896), (w, h)

    # away from a tie there is nothing to argue about
    assert _box(1000, 902, multiple_of=8)[:2] == (1000, 904)
    assert _box(1000, 899, multiple_of=8)[:2] == (1000, 896)


def test_multiple_of_off_gives_the_literal_box():
    assert _box(1001, 903, multiple_of=1)[:2] == (1001, 903)


def test_position_and_offsets_still_frame_it():
    """The box fixes the size; these still decide which part survives."""
    for pos in ["center", "top", "bottom", "left", "right"]:
        for ox, oy in [(0, 0), (200, 0), (0, 200), (-9999, 9999)]:
            got = _box(1920, 1080, crop_position=pos,
                       crop_offset_x=ox, crop_offset_y=oy)[:2]
            assert got == (1920, 1080), (pos, ox, oy, got)


def test_it_overrides_the_manual_crop_rather_than_stacking():
    """Two crops fighting over one result would be nobody's idea of clear."""
    got = _box(1920, 1080, crop=True, crop_ratio="1:1")[:2]
    assert got == (1920, 1080), got


def test_it_is_not_the_same_as_custom_crop():
    """The distinction worth being clear about.

    target_mode custom scales the image to cover the box, so it lands on those
    exact dimensions. crop_ratio custom never rescales -- it cuts the largest
    window of the shape you asked for out of what is already there. Both keep
    the shape; only the target mode guarantees the size.
    """
    assert _box(3000, 3000, src=(1024, 1536))[:2] == (3000, 3000)

    r = run(mod, image=image(1024, 1536), method="lanczos",
            target_mode="scale_factor", scale_factor=1.0, crop=True,
            crop_ratio="custom", target_width=3000, target_height=3000,
            multiple_of=1)
    got = (int(r[0].shape[2]), int(r[0].shape[1]))
    assert got == (1024, 1024), got           # the biggest square that fits


def test_latent_lands_on_the_box_in_latent_cells():
    """The box is given in image pixels, so the latent has to divide it by 8
    -- otherwise it would overshoot eightfold."""
    r = run(mod, samples=latent(128, 192), method="bicubic",
            target_mode="custom", target_width=1920, target_height=1080,
            multiple_of=1)
    lw, lh = int(r[1]["samples"].shape[3]), int(r[1]["samples"].shape[2])
    assert (lw, lh) == (240, 135), (lw, lh)


def test_image_and_latent_agree_on_the_box():
    r = run(mod, image=image(1024, 1536), samples=latent(128, 192),
            method="lanczos", target_mode="custom",
            target_width=1920, target_height=1080, multiple_of=1)
    iw, ih = int(r[0].shape[2]), int(r[0].shape[1])
    lw, lh = int(r[1]["samples"].shape[3]), int(r[1]["samples"].shape[2])
    assert (iw, ih) == (1920, 1080), (iw, ih)
    assert abs(iw / ih - lw / lh) < 0.05, (iw, ih, lw, lh)


# ---- the two crop controls that changed -----------------------------------

def test_offsets_move_on_their_own_axis():
    """One offset could only ever move along whichever axis the anchor left
    free -- with center that meant no horizontal nudge at all."""
    c = mod._crop_rect(2000, 1000, "1:1", "auto", 0, 0, "center", 0, 0)
    x = mod._crop_rect(2000, 1000, "1:1", "auto", 0, 0, "center", 300, 0)
    y = mod._crop_rect(2000, 1000, "1:1", "auto", 0, 0, "center", 0, 300)
    assert x[0] > c[0] and x[1] == c[1], (c, x)
    assert y[1] == c[1], "a 1:1 window in a 2:1 frame has no vertical slack"

    c2 = mod._crop_rect(1000, 2000, "1:1", "auto", 0, 0, "center", 0, 0)
    y2 = mod._crop_rect(1000, 2000, "1:1", "auto", 0, 0, "center", 0, 300)
    assert y2[1] > c2[1] and y2[0] == c2[0], (c2, y2)


def test_offsets_are_clamped_independently():
    for ox, oy in [(-99999, 0), (99999, 0), (0, -99999), (0, 99999)]:
        x0, y0, x1, y1 = mod._crop_rect(2000, 2000, "16:9", "landscape",
                                        0, 0, "center", ox, oy)
        assert 0 <= x0 < x1 <= 1.0001, (ox, oy, x0, x1)
        assert 0 <= y0 < y1 <= 1.0001, (ox, oy, y0, y1)


def test_auto_orientation_follows_the_source():
    """A fixed orientation cuts a thin strip out of half a mixed batch."""
    tall = mod._crop_rect(1000, 1500, "4:5", "auto", 0, 0, "center", 0, 0)
    wide = mod._crop_rect(1500, 1000, "4:5", "auto", 0, 0, "center", 0, 0)
    tall_ar = ((tall[2] - tall[0]) * 1000) / ((tall[3] - tall[1]) * 1500)
    wide_ar = ((wide[2] - wide[0]) * 1500) / ((wide[3] - wide[1]) * 1000)
    assert tall_ar < 1 < wide_ar, (tall_ar, wide_ar)


def test_auto_beats_a_fixed_orientation_on_a_mixed_batch():
    """The whole point: auto keeps more of every frame than either fixed
    setting can, because half the batch is the wrong way round for it."""
    def area(w, h, orient):
        r = mod._crop_rect(w, h, "4:5", orient, 0, 0, "center", 0, 0)
        return (r[2] - r[0]) * (r[3] - r[1])

    batch = [(1024, 1536), (1536, 1024)]
    auto = sum(area(w, h, "auto") for w, h in batch)
    for fixed in ("portrait", "landscape"):
        assert auto > sum(area(w, h, fixed) for w, h in batch), fixed


# ---- custom box mode, and the random anchor -------------------------------

def test_custom_always_crops_to_the_box():
    """Typing a resolution and not getting it back would be no use, so the
    crop is not optional here -- the crop toggle is ignored and hidden."""
    for crop in (True, False):
        r = run(mod, image=image(1024, 1536), method="lanczos",
                target_mode="custom", target_width=1920, target_height=1080,
                multiple_of=1, crop=crop)
        got = (int(r[0].shape[2]), int(r[0].shape[1]))
        assert got == (1920, 1080), (crop, got)


def test_a_box_mode_ignores_crop_ratio():
    """In a box mode the numbers you typed ARE the output size.

    Letting crop_ratio pick a different shape here meant asking for 832x1024
    and getting 832x832 back. The box wins; crop_ratio is hidden in these
    modes precisely so it cannot argue with it.
    """
    for mode in ("custom",):
        for ratio in ("1:1", "16:9", "4:5", "custom"):
            r = run(mod, image=image(1024, 1536), method="lanczos",
                    target_mode=mode, target_width=1920, target_height=1080,
                    multiple_of=1, crop=True, crop_ratio=ratio)
            got = (int(r[0].shape[2]), int(r[0].shape[1]))
            assert got == (1920, 1080), (mode, ratio, got)


def test_custom_obeys_multiple_of():
    for m in ["8", "64"]:
        r = run(mod, image=image(1024, 1536), method="lanczos",
                target_mode="custom", target_width=1000, target_height=1000,
                multiple_of=int(m))
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        assert w % int(m) == 0 and h % int(m) == 0, (m, w, h)


def test_random_is_reproducible_from_its_seed():
    """A free-running random would make the preview a lie and a result you
    liked impossible to get back."""
    def once(seed):
        r = mod._crop_rect(2000, 2000, "16:9", "landscape", 0, 0,
                           "random", 0, 0, seed)
        return tuple(round(v, 6) for v in r)
    assert once(42) == once(42)
    assert once(42) != once(43)


def test_random_stays_inside_the_image():
    for seed in range(25):
        x0, y0, x1, y1 = mod._crop_rect(1900, 1200, "4:5", "portrait",
                                        0, 0, "random", 0, 0, seed)
        assert 0 <= x0 < x1 <= 1.0001, (seed, x0, x1)
        assert 0 <= y0 < y1 <= 1.0001, (seed, y0, y1)


def test_random_actually_moves_around():
    seen = {tuple(round(v, 3) for v in
                  mod._crop_rect(2000, 2000, "16:9", "landscape",
                                 0, 0, "random", 0, 0, s)[:2])
            for s in range(30)}
    assert len(seen) > 20, f"only {len(seen)} distinct placements in 30 seeds"


def test_random_keeps_the_size_it_was_asked_for():
    base = None
    for seed in range(10):
        r = run(mod, image=image(1024, 1536), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_ratio="16:9", crop_orientation="landscape",
                crop_position="random", crop_seed=seed)
        size = (int(r[0].shape[2]), int(r[0].shape[1]))
        base = base or size
        assert size == base, (seed, size, base)


def test_random_frames_image_and_latent_alike():
    r = run(mod, image=image(1024, 1536), samples=latent(128, 192),
            method="lanczos", target_mode="scale_factor", scale_factor=2.0,
            crop=True, crop_ratio="1:1", crop_position="random", crop_seed=7,
            multiple_of=1)
    iw, ih = int(r[0].shape[2]), int(r[0].shape[1])
    lw, lh = int(r[1]["samples"].shape[3]), int(r[1]["samples"].shape[2])
    assert abs(iw / ih - lw / lh) < 0.05, (iw, ih, lw, lh)


def test_a_custom_box_too_big_keeps_its_shape():
    """It shrinks proportionally to the largest window that fits, the way the
    named formats do -- clamping each axis on its own would hand back some
    other shape entirely."""
    for cw, ch, want_ar in [(3000, 3000, 1.0), (1920, 1080, 16 / 9),
                            (1000, 4000, 0.25)]:
        r = run(mod, image=image(1024, 1536), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_ratio="custom", target_width=cw, target_height=ch,
                multiple_of=1)
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        assert w <= 2048 and h <= 3072, (cw, ch, w, h)
        assert abs(w / h - want_ar) < 0.02, (cw, ch, w / h, want_ar)


def test_a_custom_box_that_fits_is_left_alone():
    """Only an oversized box is scaled down; one that fits is taken as given."""
    r = run(mod, image=image(1024, 1536), method="lanczos",
            target_mode="scale_factor", scale_factor=2.0, crop=True,
            crop_ratio="custom", target_width=800, target_height=600,
            multiple_of=1)
    assert (int(r[0].shape[2]), int(r[0].shape[1])) == (800, 600)


def test_every_crop_takes_the_largest_window_that_fits():
    """The promise across the whole crop stage: named formats and custom boxes
    alike give you as much of the image as their shape allows."""
    for kw in [dict(crop_ratio="1:1"), dict(crop_ratio="16:9"),
               dict(crop_ratio="custom", target_width=3000, target_height=3000),
               dict(crop_ratio="custom", target_width=6000, target_height=3000)]:
        r = run(mod, image=image(1024, 1536), method="lanczos",
                target_mode="scale_factor", scale_factor=2.0, crop=True,
                crop_orientation="landscape", multiple_of=1, **kw)
        w, h = int(r[0].shape[2]), int(r[0].shape[1])
        # touching at least one edge is what "largest that fits" means
        assert w == 2048 or h == 3072, (kw, w, h)
