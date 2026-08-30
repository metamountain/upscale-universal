"""The size maths, pinned. Source is 1024x1536 throughout -- a portrait
render, which is the case the previews and defaults were designed against."""

from helpers import load_node, image, latent, run

mod = load_node()
XFAIL = set()

SRC_W, SRC_H = 1024, 1536


def _out(**kw):
    kw.setdefault("image", image(SRC_W, SRC_H))
    kw.setdefault("method", "lanczos")     # never touch a model in tests
    r = run(mod, **kw)
    return int(r[0].shape[2]), int(r[0].shape[1]), r[4]


def test_percent_and_factor_agree():
    a = _out(target_mode="scale_percent", scale_percent=200.0)[:2]
    b = _out(target_mode="scale_factor", scale_factor=2.0)[:2]
    assert a == b == (2048, 3072), (a, b)


def test_shortest_side_pins_the_short_edge():
    w, h, _ = _out(target_mode="shortest_side", shortest_side=1536)
    assert min(w, h) == 1536, (w, h)
    assert (w, h) == (1536, 2304), (w, h)


def test_longest_side_pins_the_long_edge():
    w, h, _ = _out(target_mode="longest_side", longest_side=2048, multiple_of="off")
    assert max(w, h) == 2048, (w, h)
    assert (w, h) == (1365, 2048), (w, h)


def test_megapixels_hits_the_pixel_count():
    w, h, _ = _out(target_mode="megapixels", megapixels=2.0, multiple_of="off")
    mp = w * h / 1e6
    assert 1.94 < mp < 2.06, mp


def test_aspect_is_preserved_by_every_mode():
    src = SRC_W / SRC_H
    for kw in [dict(target_mode="scale_percent", scale_percent=137.0),
               dict(target_mode="scale_factor", scale_factor=1.37),
               dict(target_mode="shortest_side", shortest_side=999),
               dict(target_mode="longest_side", longest_side=999),
               dict(target_mode="megapixels", megapixels=1.3)]:
        w, h, _ = _out(multiple_of="off", **kw)
        assert abs(w / h - src) < 0.01, (kw, w, h)


def test_every_mode_downscales_too():
    for kw in [dict(target_mode="scale_percent", scale_percent=50.0),
               dict(target_mode="scale_factor", scale_factor=0.5),
               dict(target_mode="shortest_side", shortest_side=512),
               dict(target_mode="longest_side", longest_side=768),
               dict(target_mode="megapixels", megapixels=0.4)]:
        w, h, _ = _out(**kw)
        assert w < SRC_W and h < SRC_H, (kw, w, h)


def test_multiple_of_is_always_satisfied():
    for m in ["2", "4", "8", "16", "32", "64"]:
        for mode, kw in [("scale_factor", dict(scale_factor=1.37)),
                         ("shortest_side", dict(shortest_side=999)),
                         ("megapixels", dict(megapixels=1.3))]:
            w, h, _ = _out(target_mode=mode, multiple_of=m, **kw)
            assert w % int(m) == 0 and h % int(m) == 0, (m, mode, w, h)


def test_multiple_of_off_gives_the_exact_number():
    w, h, _ = _out(target_mode="shortest_side", shortest_side=999, multiple_of="off")
    assert min(w, h) == 999, (w, h)
