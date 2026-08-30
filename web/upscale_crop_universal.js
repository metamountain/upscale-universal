import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Upscale Crop Universal -- widget visibility.
//
// The node has two stages and a lot of widgets, but only a handful matter at
// any one time: target_mode decides which size field is live, method decides
// whether the model picker is, and crop decides whether the whole second
// block exists. Showing all of them at once would make a tall node where
// most rows are inert, so everything that does not currently apply is
// hidden.
//
// LiteGraph has no show/hide API. The convention custom nodes settle on is
// to swap the widget's type to "hidden" and zero its computeSize, keeping
// the original values so it can come back -- the widget stays in the graph
// and still serializes, it just stops being drawn.

const NODE = "UpscaleCropUniversal";

const TARGET_MODES = ["scale_percent", "scale_factor", "shortest_side",
                      "longest_side", "megapixels", "width_height", "custom"];
// Both take the target_width x target_height box; only width_height crops to
// it for you, so only that one folds the crop block away.
const BOX_MODES = ["width_height", "custom"];
const METHODS = ["model", "lanczos", "bicubic", "bilinear", "nearest", "area"];
const RATIOS = ["custom", "4:5", "1:1", "4:3", "3:2", "DIN", "16:10", "16:9", "2:1", "21:9"];
const ORIENTATIONS = ["auto", "portrait", "landscape"];
const CROP_POSITIONS = ["center", "top", "bottom", "left", "right", "random"];

// The modes whose widget is simply named after them.
const SIZE_WIDGETS = ["scale_percent", "scale_factor", "shortest_side",
                      "longest_side", "megapixels"];

// One width x height box, shared by the box modes and a custom crop shape.
const BOX_WIDGETS = ["target_width", "target_height"];

// Fallbacks for the optional widgets. ComfyUI's frontend can leave an
// optional widget at null until it is touched, and a null fails server-side
// validation before run() is ever called, so this is fixed on the widget
// itself rather than in Python.
// multiple_of used to be a combo of fixed strings; a workflow saved then
// restores "8" or "off" into what is now a number field, so it is repaired
// alongside the optional widgets even though it is a required one.
// multiple_of used to be a combo of fixed strings; a workflow saved back then
// restores "8" or "off" into what is now a number field, so it is repaired
// alongside the optional widgets even though it is a required one.
const OPTIONAL_DEFAULTS = {
    multiple_of: 8,
    multiple_of: 8,
    scale_percent: 200.0,
    scale_factor: 2.0,
    shortest_side: 1536,
    longest_side: 2048,
    megapixels: 2.0,
    target_width: 1920,
    target_height: 1080,
    crop: false,
    crop_offset_x: 0,
    crop_offset_y: 0,
    crop_seed: 0,
};

const COMBO_VALUES = {
    target_mode: TARGET_MODES,
    method: METHODS,
    crop_ratio: RATIOS,
    crop_orientation: ORIENTATIONS,
    crop_position: CROP_POSITIONS,
};

// Everything the preview depends on. upscale_model is not here: swapping the
// model changes nothing the preview can show, since it draws the cached
// thumbnail rather than re-running the upscale.
const PREVIEW_WIDGETS = [
    "target_mode", "method", "multiple_of",
    "scale_percent", "scale_factor", "shortest_side", "longest_side", "megapixels",
    "target_width", "target_height",
    "crop", "crop_ratio", "crop_orientation",
    "crop_position", "crop_offset_x", "crop_offset_y", "crop_seed",
];
const PREVIEW_DEBOUNCE_MS = 120;
const PREVIEW_MAX_H = 150;

function widget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

// A workflow saved before a widget existed restores its values by position,
// so every later widget receives the wrong one -- which is how a number can
// land in a combo that only accepts strings. Anything not a valid option
// goes back to the first.
function repairCombos(node) {
    for (const [name, valid] of Object.entries(COMBO_VALUES)) {
        const w = widget(node, name);
        if (!w) continue;
        if (!valid.includes(w.value)) w.value = valid[0];
    }
}

function repairOptionals(node) {
    for (const [name, fallback] of Object.entries(OPTIONAL_DEFAULTS)) {
        const w = widget(node, name);
        if (!w) continue;
        if (typeof w.value === "string") {
            const n = w.value.trim().toLowerCase();
            w.value = (n === "off" || n === "" || Number.isNaN(Number(n)))
                ? fallback : Number(n);
        }
        if (typeof w.value === "string") {
            const t = w.value.trim().toLowerCase();
            const n = Number(t);
            w.value = (t === "off" || t === "" || Number.isNaN(n)) ? fallback : n;
        }
        if (w.value === null || w.value === undefined ||
            (typeof w.value === "number" && Number.isNaN(w.value))) {
            w.value = fallback;
        }
        if (w._ucuGuarded) continue;
        w._ucuGuarded = true;
        const orig = w.serializeValue?.bind(w);
        w.serializeValue = async (...args) => {
            let v = orig ? await orig(...args) : w.value;
            if (v === null || v === undefined ||
                (typeof v === "number" && Number.isNaN(v))) v = fallback;
            return v;
        };
    }
}

function setVisible(node, name, visible) {
    const w = widget(node, name);
    if (!w) return;
    if (visible) {
        if (w._ucuType !== undefined) {
            w.type = w._ucuType;
            w.computeSize = w._ucuComputeSize;
            w._ucuType = undefined;
        }
    } else if (w._ucuType === undefined) {
        w._ucuType = w.type;
        w._ucuComputeSize = w.computeSize;
        w.type = "hidden";
        w.computeSize = () => [0, -4];
    }
}

function updateVisibility(node) {
    const mode = widget(node, "target_mode")?.value;
    const method = widget(node, "method")?.value;
    const cropOn = widget(node, "crop")?.value === true;
    const ratio = widget(node, "crop_ratio")?.value;
    const position = widget(node, "crop_position")?.value;

    const box = BOX_MODES.includes(mode);
    const autoCrops = mode === "width_height";

    // one size field, whichever the mode is named after
    for (const name of SIZE_WIDGETS) setVisible(node, name, name === mode);
    // one box serves both the box modes and a custom-shaped crop
    const customCrop = cropOn && !autoCrops && ratio === "custom";
    for (const name of BOX_WIDGETS) setVisible(node, name, box || customCrop);

    // the model picker only exists for method = model
    setVisible(node, "upscale_model", method === "model");

    // width_height already crops to its box, so the crop block would only
    // be a second crop arguing with the first -- it folds away entirely and
    // just the framing controls stay, since those still decide what survives
    const framing = autoCrops || cropOn;
    setVisible(node, "crop", !autoCrops);
    setVisible(node, "crop_ratio", cropOn && !autoCrops);
    setVisible(node, "crop_position", framing);
    setVisible(node, "crop_offset_x", framing);
    setVisible(node, "crop_offset_y", framing);
    // the seed only means anything to the random placement
    setVisible(node, "crop_seed", framing && position === "random");
    // custom takes the box above; a named format takes an orientation
    // instead, and a square has no orientation to take
    setVisible(node, "crop_orientation",
               cropOn && !autoCrops && ratio !== "custom" && ratio !== "1:1");

    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "upscale_crop_universal.widgets",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            repairCombos(this);
            repairOptionals(this);
            updateVisibility(this);
        };

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            repairCombos(this);
            repairOptionals(this);

            // Re-evaluate whenever one of the widgets that steers the others
            // changes. crop_ratio is in here because 'custom' swaps the
            // pixel fields in for the orientation combo.
            for (const name of ["target_mode", "method", "crop", "crop_ratio", "crop_position"]) {
                const w = widget(this, name);
                if (!w) continue;
                const orig = w.callback;
                const node = this;
                w.callback = function (...args) {
                    const r = orig?.apply(this, args);
                    updateVisibility(node);
                    return r;
                };
            }

            buildPreview(this);
            updateVisibility(this);
        };
    },
});

// ---------------------------------------------------------------- preview
//
// One image with the discarded area dimmed, the way every crop tool shows it
// -- two thumbnails side by side would each be half the size and the second
// would only repeat what the frame already says.
//
// The backend hands back the cached thumbnail plus the numbers, computed by
// the same functions the node runs, so the frame here is exactly what the
// graph will cut. Nothing is re-rendered per keystroke.

function buildPreview(node) {
    const stage = document.createElement("div");
    Object.assign(stage.style, {
        position: "relative", borderRadius: "3px", overflow: "hidden",
        background: "#191b1f", margin: "0 auto", display: "none",
    });

    const img = document.createElement("img");
    Object.assign(img.style, { display: "block", width: "100%", height: "100%" });

    const window_ = document.createElement("div");
    Object.assign(window_.style, {
        position: "absolute", border: "1px solid rgba(79,195,161,.85)",
        boxShadow: "0 0 0 999px rgba(9,11,14,.68)", display: "none",
        cursor: "move", touchAction: "none",
    });
    window_.title = "Drag to place the crop";

    const size = document.createElement("span");
    Object.assign(size.style, {
        position: "absolute", left: "3px", bottom: "3px",
        font: "600 8.5px monospace", color: "#08120e",
        background: "#4fc3a1", padding: "1px 5px", borderRadius: "2px",
        whiteSpace: "nowrap",
    });
    window_.appendChild(size);

    stage.appendChild(img);
    stage.appendChild(window_);

    const holder = document.createElement("div");
    Object.assign(holder.style, { display: "flex", justifyContent: "center" });
    holder.appendChild(stage);

    const status = document.createElement("div");
    Object.assign(status.style, {
        font: "9.5px monospace", color: "#7d8592", textAlign: "center",
        padding: "3px 0", wordBreak: "break-word", lineHeight: "1.45",
    });
    status.textContent = "run the graph once to see a preview";

    const wrap = document.createElement("div");
    Object.assign(wrap.style, { display: "flex", flexDirection: "column", gap: "4px" });
    wrap.appendChild(holder);
    wrap.appendChild(status);

    let timer = null;
    let abort = null;
    let last = null;          // the last preview payload, for the drag maths

    function draw(d) {
        last = d;
        img.src = "data:image/png;base64," + d.png;

        // fit the real dimensions into the panel, capped both ways, so a
        // portrait source does not stretch the node to twice its height
        const [uw, uh] = d.up;
        const maxW = holder.clientWidth || 260;
        const fit = Math.min(maxW / uw, PREVIEW_MAX_H / uh);
        stage.style.width = Math.max(24, Math.round(uw * fit)) + "px";
        stage.style.height = Math.max(24, Math.round(uh * fit)) + "px";
        stage.style.display = "block";

        if (d.rect && d.crop) {
            const [x0, y0, x1, y1] = d.rect;
            window_.style.left = (x0 * 100).toFixed(2) + "%";
            window_.style.top = (y0 * 100).toFixed(2) + "%";
            window_.style.width = ((x1 - x0) * 100).toFixed(2) + "%";
            window_.style.height = ((y1 - y0) * 100).toFixed(2) + "%";
            window_.style.display = "block";
            size.textContent = d.crop[0] + "×" + d.crop[1];
        } else {
            window_.style.display = "none";
        }

        const bits = [
            d.src[0] + "×" + d.src[1] + " → " + uw + "×" + uh,
            d.method,
            "×" + d.factor,
        ];
        if (d.multiple_of >= 1) bits.push("/" + Math.round(d.multiple_of));
        if (d.crop) {
            let shape = d.ratio === "custom" || d.ratio === "1:1"
                ? d.ratio : d.ratio + " " + d.orientation;
            bits.push(shape + " " + d.position + " → " + d.crop[0] + "×" + d.crop[1]);
        }
        if (d.crop) bits.push("drag to place");
        status.textContent = bits.join(" | ");
    }

    // ---- drag the window ------------------------------------------------
    //
    // The anchor plus two offsets are precise but slow to aim with. Dragging
    // writes back into those same offsets rather than introducing a second
    // way of storing the position, so the widgets stay the single source of
    // truth and a dragged crop is still reproducible from the saved graph.

    function setWidget(name, value) {
        const w = widget(node, name);
        if (!w) return;
        w.value = value;
        w.callback?.(value);
    }

    function beginDrag(ev) {
        if (!last || !last.rect || !last.crop) return;
        ev.preventDefault();
        ev.stopPropagation();

        const [uw, uh] = last.up;
        const [cw, ch] = last.crop;
        const slackX = Math.max(0, uw - cw);
        const slackY = Math.max(0, uh - ch);
        if (slackX < 1 && slackY < 1) return;      // nothing to move

        // screen px -> image px
        const rect = stage.getBoundingClientRect();
        const perPxX = uw / Math.max(1, rect.width);
        const perPxY = uh / Math.max(1, rect.height);

        // A random placement has no anchor to offset from, so taking hold of
        // the window means taking manual control: it becomes a centre anchor
        // at wherever you drop it.
        let position = widget(node, "crop_position")?.value ?? "center";
        if (position === "random") {
            position = "center";
            setWidget("crop_position", "center");
        }
        const baseX = position === "left" ? 0
                    : position === "right" ? slackX : slackX / 2;
        const baseY = position === "top" ? 0
                    : position === "bottom" ? slackY : slackY / 2;

        const startX = ev.clientX, startY = ev.clientY;
        const startLeft = last.rect[0] * uw;
        const startTop = last.rect[1] * uh;
        let endX = startLeft, endY = startTop;

        function onMove(e) {
            // draw it locally while dragging -- a round trip per mouse move
            // would lag behind the cursor
            endX = Math.max(0, Math.min(slackX, startLeft + (e.clientX - startX) * perPxX));
            endY = Math.max(0, Math.min(slackY, startTop + (e.clientY - startY) * perPxY));
            window_.style.left = ((endX / uw) * 100).toFixed(3) + "%";
            window_.style.top = ((endY / uh) * 100).toFixed(3) + "%";
        }

        function onUp() {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            setWidget("crop_offset_x", Math.round(endX - baseX));
            setWidget("crop_offset_y", Math.round(endY - baseY));
            node.setDirtyCanvas(true, true);
            schedule(0);          // confirm against the backend's own maths
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
    }

    window_.addEventListener("pointerdown", beginDrag);

    // double-click puts it back to the anchor
    window_.addEventListener("dblclick", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        setWidget("crop_offset_x", 0);
        setWidget("crop_offset_y", 0);
        schedule(0);
    });

    async function fetchPreview() {
        abort?.abort();
        abort = new AbortController();
        const body = { node_id: String(node.id) };
        for (const name of PREVIEW_WIDGETS) {
            const w = widget(node, name);
            if (w) body[name] = w.value;
        }
        try {
            const res = await api.fetchApi("/upscale_crop_universal/preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: abort.signal,
            });
            const d = await res.json();
            if (!d.ok) {
                stage.style.display = "none";
                status.textContent = d.message || "no preview yet";
                return;
            }
            draw(d);
        } catch (err) {
            if (err?.name !== "AbortError") {
                status.textContent = "preview failed: " + err;
            }
        }
    }

    function schedule(delay = PREVIEW_DEBOUNCE_MS) {
        clearTimeout(timer);
        timer = setTimeout(fetchPreview, delay);
    }

    // Re-render on every widget that matters, and again once the graph runs --
    // that is when a fresh image lands in the cache.
    for (const name of PREVIEW_WIDGETS) {
        const w = widget(node, name);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (...args) {
            const r = orig?.apply(this, args);
            schedule();
            return r;
        };
    }

    const onExecuted = node.onExecuted;
    node.onExecuted = function (...args) {
        const r = onExecuted?.apply(this, args);
        schedule(0);
        return r;
    };

    node.addDOMWidget("ucu_preview", "preview", wrap, { serialize: false });
    schedule(0);
}
