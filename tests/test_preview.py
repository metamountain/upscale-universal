"""The live preview must agree with the node.

It exists to be trusted while dragging a slider, so the numbers it reports
have to be the ones the graph will actually produce -- which is only true
because both go through the same functions. These tests pin that.
"""

from helpers import load_node, image, run

mod = load_node()
XFAIL = set()

SRC_W, SRC_H = 1024, 1536


def _cached():
    """What _remember() would have stashed after one run."""
    mod._remember("t", image(SRC_W, SRC_H))
    return mod._LAST_IMAGE["t"]


def test_remember_keeps_the_true_dimensions_not_the_thumbnail_ones():
    c = _cached()
    assert (c["w"], c["h"]) == (SRC_W, SRC_H), (c["w"], c["h"])
    assert c["png"], "expected a base64 thumbnail"


def test_cache_does_not_grow_without_bound():
    mod._LAST_IMAGE.clear()
    for i in range(40):
        mod._remember(f"n{i}", image(64, 64))
    assert len(mod._LAST_IMAGE) <= 32, len(mod._LAST_IMAGE)


def test_preview_size_matches_what_the_node_produces():
    c = _cached()
    for data in [
        {"target_mode": "scale_factor", "scale_factor": 2.0},
        {"target_mode": "scale_percent", "scale_percent": 137.0},
        {"target_mode": "shortest_side", "shortest_side": 999},
        {"target_mode": "longest_side", "longest_side": 2048},
        {"target_mode": "megapixels", "megapixels": 1.3},
    ]:
        for mult in ["off", "8", "64"]:
            d = dict(data, method="lanczos", multiple_of=mult)
            p = mod._preview(d, c)
            r = run(mod, image=image(SRC_W, SRC_H), **d)
            assert p["up"] == [int(r[0].shape[2]), int(r[0].shape[1])], (d, mult, p["up"], r[0].shape)


def test_preview_crop_size_matches_what_the_node_produces():
    c = _cached()
    for ratio in ["4:5", "1:1", "16:9", "DIN", "custom"]:
        for pos in ["center", "top", "left"]:
            d = {"target_mode": "scale_factor", "scale_factor": 2.0,
                 "method": "lanczos", "multiple_of": "8", "crop": True,
                 "crop_ratio": ratio, "crop_orientation": "portrait",
                 "crop_width": 1200, "crop_height": 900,
                 "crop_position": pos, "crop_offset_x": 0, "crop_offset_y": 0}
            p = mod._preview(d, c)
            r = run(mod, image=image(SRC_W, SRC_H), **d)
            assert p["crop"] == [int(r[0].shape[2]), int(r[0].shape[1])], \
                (ratio, pos, p["crop"], r[0].shape)


def test_preview_rect_is_normalised_and_inside_the_frame():
    c = _cached()
    for pos in ["center", "top", "bottom", "left", "right"]:
        for off in [-4000, 0, 4000]:
            d = {"target_mode": "scale_factor", "scale_factor": 2.0,
                 "method": "lanczos", "multiple_of": "8", "crop": True,
                 "crop_ratio": "16:9", "crop_orientation": "landscape",
                 "crop_position": pos, "crop_offset_y": off}
            x0, y0, x1, y1 = mod._preview(d, c)["rect"]
            assert 0 <= x0 < x1 <= 1.0001, (pos, off, x0, x1)
            assert 0 <= y0 < y1 <= 1.0001, (pos, off, y0, y1)


def test_preview_reports_no_rect_when_crop_is_off():
    p = mod._preview({"target_mode": "scale_factor", "scale_factor": 2.0}, _cached())
    assert p["rect"] is None and p["crop"] is None


def test_preview_survives_rubbish_from_the_browser():
    """Widget values arrive as JSON, so anything can turn up -- nulls from
    untouched optionals, strings where numbers belong, unknown combo values."""
    c = _cached()
    for bad in [
        {"target_mode": None, "scale_factor": None, "multiple_of": None},
        {"target_mode": "nonsense", "method": 7, "multiple_of": "x"},
        {"target_mode": "scale_factor", "scale_factor": "two"},
        {"crop": True, "crop_ratio": "nope", "crop_offset_x": None, "crop_offset_y": None},
        {},
    ]:
        p = mod._preview(bad, c)
        assert p["ok"] and p["up"][0] > 0 and p["up"][1] > 0, (bad, p)
