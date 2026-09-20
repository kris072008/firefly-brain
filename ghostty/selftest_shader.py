"""Shader half of selftest.py: strip decoding at different font sizes/paddings.

Builds terminal frames the way Ghostty would draw them (default background,
strip cells as full-cell colour blocks, a block cursor whose rectangle is what
Ghostty puts in iCurrentCursor), renders them through flybrain.glsl offscreen and
checks: the strip is hidden, exactly the neurons set to level 3 light up, and the
result does not depend on cell size, padding, a lost bottom row or gamma.
"""
from __future__ import annotations

import re
from pathlib import Path

import moderngl
import numpy as np

import daemon
import preview

HERE = Path(__file__).resolve().parent
BG = np.array(preview.BG, dtype=np.uint8)
N = 400

GLSL = re.sub(r"#define LIVE_DEBUG\s+\d", "#define LIVE_DEBUG    0", (HERE / "flybrain.glsl").read_text())   # tests run without the overlay
ASPECT = float(re.search(r"const float ASPECT = ([\d.]+);", GLSL).group(1))
FIT = float(re.search(r"#define BRAIN_FIT\s+([\d.]+)", GLSL).group(1))
XY = np.load(HERE / "flybrain_subset.npz")["xy"]

_ctx = moderngl.create_standalone_context()
_buf = _ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4"))


def make(src):
    prog = _ctx.program(
        vertex_shader="#version 410 core\nin vec2 p; void main(){gl_Position=vec4(p,0,1);}",
        fragment_shader=preview.PREFIX + src)
    return prog, _ctx.vertex_array(prog, [(_buf, "2f", "p")])


_prog, _vao = make(GLSL)
_dbg = None


def cursor_rect(cfg, kind):
    """(left, bottom edge, w, h) as Ghostty reports it, for a cursor on row 3, column 5."""
    cw, ch = cfg["cw"], cfg["ch"]
    left, bottom = cfg["padL"] + 5 * cw, cfg["padT"] + 3 * ch + ch
    if kind == "bar":                      # 2px bar centred on the left cell edge
        return (left - 1, bottom, 2, ch)
    if kind == "wide":                     # block over a double-width glyph
        return (left, bottom, 2 * cw, ch)
    if kind == "underline":
        return (left, bottom + 4, cw, 3)
    return (left, bottom, cw, ch)


# Ghostty's Metal renderer outputs Display P3: sRGB colours are linearised, multiplied by
# this matrix (XYZ_DP3 * sRGB_XYZ from shaders.metal) and re-encoded. Under
# alpha-blending=linear the texture is sRGB-format and reads back linear values;
# window-colorspace=display-p3 skips the conversion.
_SRGB_XYZ = np.array([[0.4360747, 0.3850649, 0.1430804], [0.2225045, 0.7168786, 0.0606169],
                      [0.0139322, 0.0971045, 0.7141733]])
_XYZ_DP3 = np.array([[2.40414768, -0.99010704, -0.39759019], [-0.84239098, 1.79905954, 0.01597023],
                     [0.04838763, -0.09752546, 1.27393636]])
_M = _XYZ_DP3 @ _SRGB_XYZ


def _lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _enc(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


MODES = {"default (P3 conversion, native)": (True, False), "P3 conversion, linear blending": (True, True),
         "window-colorspace=display-p3": (False, False), "display-p3 + linear blending": (False, True)}


def ghostty_view(img, converted=True, linear=False):
    """What the shader's iChannel0 returns for a terminal image drawn with sRGB colours."""
    c = img.astype(np.float64) / 255.0
    l = _lin(c)
    if converted:
        l = l @ _M.T
    q = np.round(_enc(l) * 255.0) / 255.0                 # 8-bit render target
    return _lin(q).astype(np.float32) if linear else np.round(q * 255).astype(np.uint8)


def frame(cfg, cells, mode=(True, False), cols=100, rows=24):
    """Terminal image (H, W, 3) with `cells` painted on the last row + cursor rect."""
    cw, ch = cfg["cw"], cfg["ch"]
    W = cfg["padL"] + cols * cw + cfg["remx"] + 4
    H = cfg["padT"] + rows * ch + cfg["remy"] + cfg["padB"]
    img = np.tile(BG, (H, W, 1))
    if cells is not None:
        y = cfg["padT"] + (rows - 1) * ch
        for x, c in enumerate(cells[: cols - 1]):
            px = cfg["padL"] + x * cw
            img[y:y + ch, px:px + cw] = c
    return ghostty_view(img, *mode), cursor_rect(cfg, "block")


def render(img, cursor, t=2.0, style=0, prog=None):
    _p, _v = prog or (_prog, _vao)
    H, W, _ = img.shape
    tex = _ctx.texture((W, H), 3, np.ascontiguousarray(img).tobytes(),
                       dtype="f4" if img.dtype == np.float32 else "f1")
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.use(0)
    fbo = _ctx.simple_framebuffer((W, H))
    fbo.use()
    fbo.clear(0, 0, 0, 1)

    def u(name, val):
        if name in _p:
            _p[name].value = val

    u("iResolution", (W, H, 1.0))
    u("iTime", t)
    u("iChannel0", 0)
    u("iCurrentCursorStyle", style)
    u("iCurrentCursor", (*[float(v) for v in cursor],))
    u("iPreviousCursor", (*[float(v) for v in cursor],))
    u("iTimeCursorChange", -100.0)
    u("iBackgroundColor", (0.0, 0.0, 0.0))
    _v.render()
    out = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)[::-1].astype(float)
    tex.release()
    fbo.release()
    return out


def neuron_px(j, W, H):
    scale = min(W, H / ASPECT) * FIT
    return XY[j, 0] * scale + W / 2, XY[j, 1] * scale + H / 2


def patch_luma(out, j, r=6):
    x, y = neuron_px(j, out.shape[1], out.shape[0])
    x, y = int(round(x)), int(round(y))
    return out[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1].sum(axis=2).mean()


CONFIGS = {
    "13pt retina (16x34, pad 4)": dict(cw=16, ch=34, padL=4, padT=4, remx=7, remy=9, padB=4),
    "small font (10x22)": dict(cw=10, ch=22, padL=4, padT=4, remx=3, remy=5, padB=4),
    "large font (25x51)": dict(cw=25, ch=51, padL=4, padT=4, remx=11, remy=17, padB=4),
    "balanced padding (pad 11/13)": dict(cw=16, ch=34, padL=11, padT=13, remx=3, remy=2, padB=13),
    "bottom padding costs a row": dict(cw=16, ch=34, padL=4, padT=4, remx=0, remy=32, padB=4),
}


def run(check):
    print("shader (decode at different geometries)")
    lit = [24, 115, 260, 303, 376, 359]   # one neuron in each of the six slots of a cell (j % 6 = 0..5)
    lv = np.zeros(N, dtype=np.uint8)
    lv[lit] = 3
    cells_on, cells_off = daemon.encode_cells(lv), daemon.encode_cells(np.zeros(N, dtype=np.uint8))
    others = [j for j in range(0, N, 9) if j not in lit and abs(neuron_px(j, 1600, 900)[0] - 800) < 700][:30]

    for name, cfg in CONFIGS.items():
        for mname, mode in MODES.items():
            img_on, cur = frame(cfg, cells_on, mode)
            img_off, _ = frame(cfg, cells_off, mode)
            on, off = render(img_on, cur), render(img_off, cur)
            H, W, _ = on.shape
            rows = 24
            y_strip = cfg["padT"] + (rows - 1) * cfg["ch"] + cfg["ch"] // 2
            hidden = np.abs(on[y_strip, 8:W - 8] - BG).max() < 90       # glow only, no magic colours
            gain = min(patch_luma(on, j) - patch_luma(off, j) for j in lit)
            leak = np.mean([abs(patch_luma(on, j) - patch_luma(off, j)) for j in others])
            tag = f"{name}, {mname}"
            check(f"{tag}: strip hidden", hidden)
            check(f"{tag}: level-3 neurons light up (weakest +{gain:.0f} luma) without touching others (leak {leak:.1f})",
                  gain > 25 and leak < 4)

    # cursor styles: the prompt cursor under Ghostty's zsh integration is a BAR
    cfg = CONFIGS["13pt retina (16x34, pad 4)"]
    img_on, _ = frame(cfg, cells_on)
    img_off, _ = frame(cfg, cells_off)
    y_strip = cfg["padT"] + 23 * cfg["ch"] + cfg["ch"] // 2
    for kind, style in (("block", 0), ("hollow", 1), ("bar", 2), ("wide", 0)):
        cur = cursor_rect(cfg, kind)
        on, off = render(img_on, cur, style=style), render(img_off, cur, style=style)
        hidden = np.abs(on[y_strip, 8:on.shape[1] - 8] - BG).max() < 90
        gain = min(patch_luma(on, j) - patch_luma(off, j) for j in lit)
        check(f"{kind} cursor: strip hidden and decoded (weakest +{gain:.0f} luma)", hidden and gain > 25)

    # no daemon: geometry must not be found and nothing crashes
    img, cur = frame(cfg, None)
    out = render(img, cur)
    check("without a strip the shader falls back to the baked replay", out.max() > 60)

    # underline / missing cursor: cannot locate the row (documented limitation)
    out_u = render(img_on, cursor_rect(cfg, "underline"), style=3)
    check("underline cursor: declined without crashing (strip stays visible)",
          np.abs(out_u[y_strip, 40:200] - BG).max() > 60)

    # debug indicator: green = decoded, orange = hidden but not decoded, red = no strip
    dbg = make(GLSL.replace("#define LIVE_DEBUG    0", "#define LIVE_DEBUG    3"))

    def corner(o):
        return o[18, o.shape[1] - 18]
    live = corner(render(img_on, cursor_rect(cfg, "block"), prog=dbg))
    check("debug indicator is green when decoding", live[1] > 200 and live[0] < 60, f"{live}")
    none = corner(render(frame(cfg, None)[0], cursor_rect(cfg, "block"), prog=dbg))
    check("debug indicator is red with no strip", none[0] > 200 and none[1] < 60, f"{none}")
    # strip present, magic header damaged (row still found via cell 0, but cells unreadable)
    bad = img_on.copy()
    x1 = cfg["padL"] + int(1.5 * cfg["cw"])
    y1 = cfg["padT"] + 23 * cfg["ch"]
    bad[y1:y1 + cfg["ch"], x1 - 4:x1 + 4] = (90, 90, 200)
    part = corner(render(bad, cursor_rect(cfg, "block"), prog=dbg))
    check("debug indicator is orange when hidden but not decodable", part[0] > 200 and 100 < part[1] < 200, f"{part}")
