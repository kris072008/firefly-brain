"""Render flybrain.glsl offscreen with a fake terminal on top, to PNG.

    pip install moderngl pillow
    python ghostty/preview.py --times 1 4 8 --out /tmp/fb        # PNG stills
    python ghostty/preview.py --bench                             # GPU ms/frame

Mimics Ghostty's shader wrapper on Metal: y-down fragCoord, iChannel0 holding
the rendered terminal, iBackgroundColor = the default #282c34 background.
Cursor uniforms can be set with --cursor X Y (pixels, y-down).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BG = (0x28, 0x2C, 0x34)

PREFIX = """#version 410 core
uniform vec3  iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int   iFrame;
uniform vec4  iCurrentCursor;
uniform vec4  iPreviousCursor;
uniform vec4  iCurrentCursorColor;
uniform int   iCursorVisible;
uniform float iTimeCursorChange;
uniform vec3  iBackgroundColor;
uniform vec3  iForegroundColor;
uniform sampler2D iChannel0;
out vec4 _fragColor;
void mainImage( out vec4 fragColor, in vec2 fragCoord );
void main() { mainImage(_fragColor, vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y)); }
"""

TEXT = """kris@air ~/flyweb % ls -la
total 96
drwxr-xr-x@  10 kris  staff    320 19 Sep 11:33 .
-rw-r--r--@   1 kris  staff   6148 18 Sep 02:41 .DS_Store
drwxr-xr-x@  17 kris  staff    544 19 Sep 11:19 flybrain
-rw-r--r--@   1 kris  staff   5819 19 Sep 11:19 README.md
kris@air ~/flyweb % git log --oneline
4a69a03 add .gitignore, untrack cache and .DS_Store files
3752ebe initial commit
kris@air ~/flyweb % python -m flybrain check
one synapse produces 0.0433 mV   (analytic 0.0433 mV)
synapses needed to reach threshold: 160
PASS
def step(self, stim=None, rng=None):
    arriving = self._buf[self._cursor]
    if arriving.size:
        self.i_syn += np.asarray(self.W[arriving].sum(axis=0)).ravel()
kris@air ~/flyweb % """


def fake_terminal(w, h, cursor_xy=None):
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 26)
    colors = [(255, 255, 255), (171, 178, 191), (152, 195, 121), (97, 175, 239)]
    y = 40
    for i, line in enumerate(TEXT.split("\n")):
        d.text((40, y), line, font=font, fill=colors[i % 4])
        y += 36
    if cursor_xy:
        d.rectangle([cursor_xy[0], cursor_xy[1] - 34, cursor_xy[0] + 16, cursor_xy[1]],
                    fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader", default=str(HERE / "flybrain.glsl"))
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 1000])
    ap.add_argument("--times", type=float, nargs="*", default=[1.0])
    ap.add_argument("--out", default="preview")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--zero-bg-uniform", action="store_true",
                    help="simulate Ghostty leaving iBackgroundColor at 0")
    ap.add_argument("--cursor", type=float, nargs=2, default=None)
    ap.add_argument("--cursor-age", type=float, default=0.3)
    args = ap.parse_args()
    w, h = args.size

    ctx = moderngl.create_standalone_context()
    src = PREFIX + Path(args.shader).read_text()
    prog = ctx.program(
        vertex_shader="#version 410 core\nin vec2 p; void main(){gl_Position=vec4(p,0,1);}",
        fragment_shader=src)
    vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4"))
    vao = ctx.vertex_array(prog, [(vbo, "2f", "p")])

    term = Image.new("RGB", (w, h), BG) if args.no_text else fake_terminal(
        w, h, tuple(args.cursor) if args.cursor else None)
    tex = ctx.texture((w, h), 3, term.tobytes())  # rows top-first == y-down
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.use(0)
    fbo = ctx.simple_framebuffer((w, h))
    fbo.use()

    def set_u(name, val):
        if name in prog:
            prog[name].value = val

    set_u("iResolution", (w, h, 1.0))
    set_u("iBackgroundColor", (0.0, 0.0, 0.0) if args.zero_bg_uniform else tuple(c / 255 for c in BG))
    set_u("iForegroundColor", (1.0, 1.0, 1.0))
    set_u("iChannel0", 0)
    set_u("iCursorVisible", 1)

    def render(t):
        set_u("iTime", t)
        if args.cursor:
            cx, cy = args.cursor
            set_u("iCurrentCursor", (cx, cy, 16.0, 34.0))
            set_u("iPreviousCursor", (cx - 16, cy, 16.0, 34.0))
            set_u("iTimeCursorChange", t - args.cursor_age)
        vao.render()

    if args.bench:
        for _ in range(5):
            render(1.0)
        ctx.finish()
        n = 60
        t0 = time.time()
        for i in range(n):
            render(1.0 + i * 0.05)
            fbo.read(components=1, alignment=1)[:1]   # force sync
        dt = (time.time() - t0) / n
        print(f"{w}x{h}: {dt * 1000:.2f} ms/frame  ({1 / dt:.0f} fps ceiling)")
        return

    for t in args.times:
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        render(t)
        raw = fbo.read(components=3)
        img = Image.frombytes("RGB", (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM)
        path = f"{args.out}_{t:05.2f}.png"
        img.save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
