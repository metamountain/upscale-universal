import { app } from "../../scripts/app.js";

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
                      "longest_side", "megapixels"];
const METHODS = ["model", "lanczos", "bicubic", "bilinear", "nearest", "area"];
const RATIOS = ["custom", "4:5", "1:1", "4:3", "3:2", "DIN", "16:10", "16:9", "2:1", "21:9"];
const ORIENTATIONS = ["portrait", "landscape"];
const CROP_POSITIONS = ["center", "top", "bottom", "left", "right"];
const MULTIPLES = ["off", "2", "4", "8", "16", "32", "64"];

// Which size widget belongs to which target_mode.
const MODE_WIDGET = {
    scale_percent: "scale_percent",
    scale_factor: "scale_factor",
    shortest_side: "shortest_side",
    longest_side: "longest_side",
    megapixels: "megapixels",
};

// Fallbacks for the optional widgets. ComfyUI's frontend can leave an
// optional widget at null until it is touched, and a null fails server-side
// validation before run() is ever called, so this is fixed on the widget
// itself rather than in Python.
const OPTIONAL_DEFAULTS = {
    scale_percent: 200.0,
    scale_factor: 2.0,
    shortest_side: 1536,
    longest_side: 2048,
    megapixels: 2.0,
    crop: false,
    crop_width: 1024,
    crop_height: 1024,
    crop_offset: 0,
};

const COMBO_VALUES = {
    target_mode: TARGET_MODES,
    method: METHODS,
    multiple_of: MULTIPLES,
    crop_ratio: RATIOS,
    crop_orientation: ORIENTATIONS,
    crop_position: CROP_POSITIONS,
};

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

    // one size field, whichever the mode asks for
    for (const name of Object.values(MODE_WIDGET)) {
        setVisible(node, name, MODE_WIDGET[mode] === name);
    }

    // the model picker only exists for method = model
    setVisible(node, "upscale_model", method === "model");

    // the whole crop block folds away when it is off
    setVisible(node, "crop_ratio", cropOn);
    setVisible(node, "crop_position", cropOn);
    setVisible(node, "crop_offset", cropOn);
    // custom takes pixels; a named format takes an orientation instead,
    // and a square has no orientation to take
    setVisible(node, "crop_width", cropOn && ratio === "custom");
    setVisible(node, "crop_height", cropOn && ratio === "custom");
    setVisible(node, "crop_orientation", cropOn && ratio !== "custom" && ratio !== "1:1");

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
            for (const name of ["target_mode", "method", "crop", "crop_ratio"]) {
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

            updateVisibility(this);
        };
    },
});
