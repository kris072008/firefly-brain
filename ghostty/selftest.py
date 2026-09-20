"""Headless checks for the live bridge (no Ghostty needed).

    pip install pyte moderngl pillow
    python ghostty/selftest.py              # codec + proxy
    python ghostty/selftest.py --shader     # also render through flybrain.glsl

codec   encode/decode round trip, including colour noise
proxy   runs daemon.py inside a pty emulated by pyte: the shell must work, keep
        out of the last row through heavy scrolling and a resize, and the strip
        must decode after the resize
shader  feeds the emulated screen to flybrain.glsl at several font sizes and
        checks that the strip is hidden and the right neurons light up
"""
from __future__ import annotations

import argparse
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import time
from pathlib import Path

import numpy as np
import pyte

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import daemon  # noqa: E402

N = 400


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name} {detail}")
    if not ok:
        check.failed += 1


check.failed = 0


# ---- codec ------------------------------------------------------------------
def test_codec():
    print("codec")
    rng = np.random.default_rng(0)
    lv = rng.integers(0, 4, N).astype(np.uint8)
    cells = daemon.encode_cells(lv)
    check("round trip", (daemon.decode_cells(cells, N) == lv).all(), f"({len(cells)} cells)")
    noisy = [tuple(int(np.clip(c + rng.integers(-6, 7), 0, 255)) for c in cell) for cell in cells]
    check("survives +-6 colour noise", (daemon.decode_cells(noisy, N) == lv).all())
    check("fits 80 columns", len(cells) <= 79, f"(needs {len(cells) + 1} columns)")


# ---- terminal emulation harness --------------------------------------------
class Term:
    """Runs a command in a pty and mirrors what it draws into a pyte screen."""

    def __init__(self, argv, rows, cols):
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.ByteStream(self.screen)
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execvp(argv[0], argv)
        self.resize(rows, cols)

    def resize(self, rows, cols, px=(16, 32)):
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, cols * px[0], rows * px[1]))
        self.screen.resize(rows, cols)

    def send(self, s):
        os.write(self.fd, s.encode())

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.05)
            if r:
                try:
                    data = os.read(self.fd, 65536)
                except OSError:
                    return
                self.stream.feed(data)

    def strip_cells(self):
        row = self.screen.buffer[self.screen.lines - 1]
        cells = []
        for x in range(self.screen.columns):
            bg = row[x].bg
            cells.append(tuple(int(bg[i:i + 2], 16) for i in (0, 2, 4)) if len(bg) == 6 else None)
        return cells

    def text(self, y):
        return "".join(self.screen.buffer[y][x].data for x in range(self.screen.columns)).rstrip()

    def close(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
            os.waitpid(self.pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def strip_ok(t):
    cells = t.strip_cells()
    if cells[0] != daemon.MAGIC_A or cells[1] != daemon.MAGIC_B or cells[2] != daemon.GAMMA_CAL:
        return False, "header missing"
    n_cells = daemon.HEADER_CELLS + -(-N // daemon.NEURONS_PER_CELL)
    room = t.screen.columns - 1
    if room < n_cells:
        return True, f"(narrow window: {room - daemon.HEADER_CELLS} of {n_cells - daemon.HEADER_CELLS} payload cells)"
    if any(c is None for c in cells[:n_cells]):
        return False, "gap in payload"
    daemon.decode_cells(cells[:n_cells], N)
    return True, f"({n_cells} cells)"


def test_proxy():
    print("proxy")
    rows, cols = 24, 100
    t = Term([sys.executable, str(HERE / "daemon.py"), "--pattern", "--shell", "/bin/sh"], rows, cols)
    try:
        t.pump(1.0)
        ok, d = strip_ok(t)
        check("strip painted on last row", ok, d)
        t.send("echo hello-from-shell\n")
        t.pump(0.5)
        check("shell works through the proxy", any("hello-from-shell" in t.text(y) for y in range(rows)))

        t.send("seq 1 300\n")
        t.pump(1.0)
        ok, d = strip_ok(t)
        check("strip survives 300 lines of scrolling", ok, d)
        body = [t.text(y) for y in range(rows - 1)]
        check("output stays above the strip (last body row filled, strip row has no text)",
              any("300" in b or "299" in b for b in body) and t.text(rows - 1) == "")

        t.send("sleep 30\n")
        t.pump(0.4)
        t.send("\x03")
        t.pump(0.4)
        t.send("echo after-interrupt\n")
        t.pump(0.5)
        check("Ctrl-C reaches the foreground job through the proxy",
              any("after-interrupt" in t.text(y) for y in range(rows)))

        t.resize(30, 120)
        os.kill(t.pid, signal.SIGWINCH)
        t.pump(1.0)
        ok, d = strip_ok(t)
        check("strip repainted on the new last row after growing to 30x120", ok, d)
        t.send("stty size\n")
        t.pump(0.5)
        check("shell sees one row less than the window (29 120)",
              any(t.text(y).strip() == "29 120" for y in range(30)))

        t.resize(20, 60)
        os.kill(t.pid, signal.SIGWINCH)
        t.pump(1.0)
        ok, d = strip_ok(t)
        check("narrow 20x60 window still paints a valid header", ok, d)

        t.send("printf '\\033[?1049h\\033[2J\\033[1;1Halt-screen'; sleep 0.3; printf '\\033[?1049l'\n")
        t.pump(1.0)
        ok, d = strip_ok(t)
        check("strip survives an alternate-screen app", ok, d)
        t.send("exit\n")
        t.pump(1.0)
        try:
            done = os.waitpid(t.pid, os.WNOHANG)[0] == t.pid
        except ChildProcessError:
            done = True
        check("daemon exits when the shell exits", done)
        check("strip is cleared on exit", t.strip_cells()[0] != daemon.MAGIC_A)
    finally:
        t.close()


def test_nested_guard():
    print("nesting")
    import subprocess
    env = dict(os.environ, FLYBRAIN_BRIDGE="4242")
    # needs a tty on stdin/stdout, so run it in a pty
    t = Term.__new__(Term)
    t.screen = pyte.Screen(100, 24)
    t.stream = pyte.ByteStream(t.screen)
    t.pid, t.fd = pty.fork()
    if t.pid == 0:
        os.environ["FLYBRAIN_BRIDGE"] = "4242"
        os.execvp(sys.executable, [sys.executable, str(HERE / "daemon.py"), "--pattern"])
    t.resize(24, 100)
    try:
        t.pump(1.5)
        text = "\n".join(t.text(y) for y in range(24))
        check("typed inside a bridge: refuses and explains (no silent nested shell)",
              "already running inside a bridge" in text and "4242" in text)
    finally:
        t.close()


def test_start_at_bottom():
    print("startup")
    rows, cols = 24, 100
    t = Term.__new__(Term)
    t.screen = pyte.Screen(cols, rows)
    t.stream = pyte.ByteStream(t.screen)
    t.stream.feed(b"".join(b"old line %d\r\n" % i for i in range(40)))     # cursor ends on the bottom row
    t.pid, t.fd = pty.fork()
    if t.pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.execvp(sys.executable, [sys.executable, str(HERE / "daemon.py"), "--pattern", "--shell", "/bin/sh"])
    t.resize(rows, cols)
    try:
        # answer the daemon's cursor-position query the way a real terminal does
        end = time.time() + 1.5
        while time.time() < end:
            r, _, _ = select.select([t.fd], [], [], 0.05)
            if r:
                data = os.read(t.fd, 65536)
                if b"\x1b[6n" in data:
                    os.write(t.fd, b"\x1b[%d;1R" % rows)
                t.stream.feed(data)
        t.send("echo first\necho second\n")
        t.pump(0.6)
        ok, d = strip_ok(t)
        check("strip intact when the terminal cursor started on the bottom row", ok, d)
        body = [t.text(y) for y in range(rows - 1)]
        check("shell output went above the strip", any("second" in b for b in body) and t.text(rows - 1) == "")
    finally:
        t.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shader", action="store_true")
    args = ap.parse_args()
    test_codec()
    test_proxy()
    test_start_at_bottom()
    test_nested_guard()
    if args.shader:
        import selftest_shader
        selftest_shader.run(check)
    print("FAILED" if check.failed else "all checks passed")
    sys.exit(1 if check.failed else 0)
