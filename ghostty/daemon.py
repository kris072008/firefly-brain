"""Live fly-brain bridge for the Ghostty shader.

    python ghostty/daemon.py                 # run your shell inside the bridge
    exec python ghostty/daemon.py --rc       # ... from the end of ~/.zshrc, if you like

Ghostty shaders cannot read files or sockets; the only thing they can see is
what is drawn in the terminal. So this program hosts your shell on a private
pty one row shorter than the window and uses the freed last row as a data strip:
each cell is a truecolor background block whose RGB bytes carry the live
activity of the 400 neurons the shader draws. The shader decodes the row from
iChannel0 and hides it.

Processes
  proxy  (this one)  relays keys/output between Ghostty and the shell, keeps the
                     shell out of the last row (pty size + scroll region), tracks
                     window resizes, and paints the strip ~30 times a second.
  sim    (child)     the flyweb LIF brain, stepped at --speed x real time, driven
                     by your typing (mechanosensory), your terminal output
                     (visual) and a slow cycle of taste/sight/touch/smell scenes.
                     It writes one activity level per shown neuron into shared
                     memory.

Strip format (cell = one terminal column, x = column index)
  cell 0   (255,   0, 255)  magic A
  cell 1   (  0, 255,   0)  magic B
  cell 2   (128, 128, 128)  calibration grey: with the two magic cells it tells the shader
                            how Ghostty's render target altered the colours (Ghostty
                            converts sRGB to Display P3, and can hand the shader linear
                            values), so the payload can be un-converted exactly
  cell 3+  payload, NEURONS_PER_CELL neurons per cell. Channel k of a cell holds
           neurons 6c+2k (low 2 bits) and 6c+2k+1 (high 2 bits) as a 4-bit value
           v, stored as the byte 17*v. Each neuron's level 0..3 is a decaying
           spike trace (0 silent, 3 just fired).
The shader finds the row from the cursor rectangle and confirms it by the magic
colour in cell 0; the cell width is the block cursor's width or, for a bar cursor,
measured from cell 0's edges. Nothing depends on font size, DPI or window padding.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import select
import signal
import struct
import sys
import termios
import time
import tty
from multiprocessing import get_context, shared_memory
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MAGIC_A = (255, 0, 255)
MAGIC_B = (0, 255, 0)
GAMMA_CAL = (128, 128, 128)
HEADER_CELLS = 3
NEURONS_PER_CELL = 6

# shared memory layout
SHM_FRAME, SHM_KEYS, SHM_OUT, SHM_READY, SHM_LEVELS = 0, 8, 12, 16, 64


# ---- codec ------------------------------------------------------------------
def encode_cells(levels) -> list[tuple[int, int, int]]:
    levels = np.asarray(levels, dtype=np.uint8) & 3
    n = len(levels)
    pad = (-n) % NEURONS_PER_CELL
    lv = np.concatenate([levels, np.zeros(pad, dtype=np.uint8)]).reshape(-1, 3, 2)
    v = lv[:, :, 0] | (lv[:, :, 1] << 2)                    # (cells, 3 channels) 0..15
    payload = [tuple(int(b) for b in row) for row in (v * 17)]
    return [MAGIC_A, MAGIC_B, GAMMA_CAL] + payload


def decode_cells(cells, n: int) -> np.ndarray:
    """Inverse of encode_cells, tolerant to +-8 counts of colour noise."""
    out = np.zeros(n, dtype=np.uint8)
    for j in range(n):
        c, r = divmod(j, NEURONS_PER_CELL)
        chan, hi = divmod(r, 2)
        v = int(round(cells[HEADER_CELLS + c][chan] / 17.0))
        out[j] = (v >> 2) & 3 if hi else v & 3
    return out


def paint_bytes(cells, rows: int, cols: int, reset_region: bool) -> bytes:
    """One atomic write that paints the last row and puts the cursor back."""
    cells = cells[: max(cols - 1, 0)]                        # never touch the last column
    out = [b"\x1b7", b"\x1b[?6l"]                            # save cursor, origin mode off
    if reset_region:
        out.append(b"\x1b[1;%dr" % (rows - 1))               # scroll region excludes the strip
    out.append(b"\x1b[%d;1H\x1b[0m" % rows)
    out.extend(b"\x1b[48;2;%d;%d;%dm " % c for c in cells)
    out.append(b"\x1b[0m\x1b[K\x1b8")                        # clear rest of row, restore cursor
    return b"".join(out)


# ---- terminal helpers -------------------------------------------------------
def get_winsize(fd):
    rows, cols, xp, yp = struct.unpack("HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\0" * 8))
    return rows, cols, xp, yp


def set_winsize(fd, rows, cols, xp, yp):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, xp, yp))


def cursor_row(timeout=0.4):
    """Ask the terminal where the cursor is. Returns (row or None, other bytes typed meanwhile)."""
    os.write(1, b"\x1b[6n")
    buf, end = b"", time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([0], [], [], 0.05)
        if r:
            buf += os.read(0, 256)
            m = re.search(rb"\x1b\[(\d+);(\d+)R", buf)
            if m:
                return int(m.group(1)), buf[:m.start()] + buf[m.end():]
    return None, buf


REGION_TRIGGERS = (b"\x1b[r", b"\x1b[;r", b"\x1bc", b"\x1b[!p",
                   b"\x1b[?1049", b"\x1b[?1047", b"\x1b[?47")


# ---- simulation process -----------------------------------------------------
def sim_main(shm_name: str, speed: float, fps: float, subset_path: str, seed: int):
    sys.path.insert(0, str(ROOT))
    from flybrain.brain import Brain
    from flybrain.data import load

    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray((1,), np.uint64, shm.buf, SHM_FRAME)
    keys = np.ndarray((1,), np.float32, shm.buf, SHM_KEYS)
    outr = np.ndarray((1,), np.float32, shm.buf, SHM_OUT)
    ready = np.ndarray((1,), np.uint32, shm.buf, SHM_READY)

    sel = np.load(subset_path)["sel"].astype(np.int64)
    n_shown = len(sel)
    levels = np.ndarray((n_shown,), np.uint8, shm.buf, SHM_LEVELS)

    ann, W, _ = load()
    brain = Brain(ann, W)
    rng = np.random.default_rng(seed)

    # name, neurons driven, peak rate (Hz), driven fraction of the scene, scene seconds.
    # Tuned against the model, which is bistable: taste/touch/sight respond while
    # driven and fall silent afterwards, but smell input (or any stray spike among
    # the strongly wired hub neurons once the network is primed) tips the brain
    # into a self-sustaining storm at any strength. So each scene starts from a
    # reset brain, smell is a short burst that is quenched mid-scene, and a
    # watchdog resets any storm that outlasts its input.
    scenes = [("taste", brain.select(cell_sub_class="sugar/water"), 250.0, 1.0, 8.0),
              ("sight", brain.select(cell_class="visual"), 100.0, 1.0, 9.0),
              ("touch", brain.select(cell_class="mechanosensory"), 30.0, 1.0, 9.0),
              ("smell", brain.select(cell_class="olfactory"), 8.0, 0.3, 6.0)]
    groups = {name: idx for name, idx, *_ in scenes}
    idx_all = np.concatenate([g for g in groups.values()]).astype(np.int32)
    offs, o = {}, 0
    for name, g in groups.items():
        offs[name] = (o, o + len(g))
        o += len(g)
    rate = np.zeros(len(idx_all), dtype=np.float32)

    lookup = np.full(brain.n, -1, dtype=np.int32)
    lookup[sel] = np.arange(n_shown, dtype=np.int32)
    trace = np.zeros(n_shown, dtype=np.float32)
    dt = brain.p.dt
    tau_wall = 0.35
    decay = np.float32(np.exp(-dt / (tau_wall * speed)))

    starts = np.cumsum([0.0] + [sc[4] for sc in scenes])
    cycle, t_start = float(starts[-1]), time.perf_counter()

    def drive(now):
        """Set the Poisson drive for this instant; returns the total sensory drive."""
        t = now - t_start
        tt = t % cycle
        k = int(np.searchsorted(starts, tt, side="right")) - 1
        ph = (tt - starts[k]) / scenes[k][4]
        rate[:] = 0.0
        name, _, peak, frac, _ = scenes[k]
        if ph < frac:
            a, b = offs[name]
            rate[a:b] += peak * np.sin(np.pi * ph / frac) ** 2
        a, b = offs["touch"]
        rate[a:b] += min(45.0, 8.0 * float(keys[0]))          # your typing
        a, b = offs["sight"]
        rate[a:b] += min(80.0, 12.0 * np.log1p(float(outr[0]) / 200.0))   # your output
        return float(rate.sum()), k, ph

    ready[0] = 1
    t0 = time.perf_counter()
    sim_t = 0.0
    next_out = next_drive = t0
    spikes_ema, quiet_since = 0.0, None
    last_k, quenched = -1, False
    while True:
        now = time.perf_counter()
        if now >= next_drive:
            sensory, k, ph = drive(now)
            next_drive = now + 0.05
            if k != last_k or (scenes[k][0] == "smell" and ph >= 0.5 and not quenched):
                quenched = k == last_k                            # reset at scene start, and once mid-smell
                brain.reset()
                trace[:] = 0
                spikes_ema, quiet_since = 0.0, None
                last_k = k
            # watchdog: brain-wide spiking that outlasts its input is a runaway
            if sensory < 50.0 and spikes_ema > 100.0:
                quiet_since = quiet_since or now
                if now - quiet_since > 0.5:
                    brain.reset()
                    trace[:] = 0
                    spikes_ema, quiet_since = 0.0, None
            else:
                quiet_since = None
        behind = (now - t0) * speed - sim_t
        if behind > 0.25 * speed:                             # cannot keep up: slow down
            t0 += (behind - 0.05 * speed) / speed
            behind = 0.05 * speed
        for _ in range(min(int(behind / dt), 250)):
            fired = brain.step((idx_all, rate), rng)
            spikes_ema += 0.02 * (fired.size - spikes_ema)
            trace *= decay
            if fired.size:
                pos = lookup[fired]
                trace[pos[pos >= 0]] = 1.0
            sim_t += dt
        if now >= next_out:
            q = np.zeros(n_shown, dtype=np.uint8)
            q[trace > 0.08] = 1
            q[trace > 0.33] = 2
            q[trace > 0.66] = 3
            levels[:] = q
            frame[0] += 1
            next_out = now + 1.0 / fps
        time.sleep(0.002)


# ---- proxy ------------------------------------------------------------------
def run(args) -> int:
    if not (os.isatty(0) and os.isatty(1)):
        print("flybrain daemon needs a terminal on stdin and stdout", file=sys.stderr)
        return 2
    n_shown = int(len(np.load(args.subset)["sel"])) if Path(args.subset).exists() else 400
    shell = args.shell or os.environ.get("SHELL", "/bin/zsh")

    # A shell inside a bridge must not start a second one. From .zshrc (--rc) that
    # is expected and silent; typed by hand it means you are already in a bridge.
    if os.environ.get("FLYBRAIN_BRIDGE"):
        if args.rc:
            os.execvp(shell, [shell])
        print("flybrain: this shell is already running inside a bridge "
              f"(daemon pid {os.environ['FLYBRAIN_BRIDGE']}); nothing started.\n"
              "  Leave it first (exit / Ctrl-D) and start the daemon from the outer shell,\n"
              "  or use another Ghostty window/tab.", file=sys.stderr)
        return 1
    os.environ["FLYBRAIN_BRIDGE"] = str(os.getpid())

    shm = shared_memory.SharedMemory(create=True, size=SHM_LEVELS + max(n_shown, 1024))
    shm.buf[:SHM_LEVELS + n_shown] = b"\0" * (SHM_LEVELS + n_shown)
    keys = np.ndarray((1,), np.float32, shm.buf, SHM_KEYS)
    outr = np.ndarray((1,), np.float32, shm.buf, SHM_OUT)
    frame = np.ndarray((1,), np.uint64, shm.buf, SHM_FRAME)
    levels = np.ndarray((n_shown,), np.uint8, shm.buf, SHM_LEVELS)
    ready = np.ndarray((1,), np.uint32, shm.buf, SHM_READY)

    sim = None
    if not args.pattern:
        ctx = get_context("spawn")
        sim = ctx.Process(target=sim_main, daemon=True,
                          args=(shm.name, args.speed, args.fps, args.subset, 0))
        sim.start()

    rows, cols, xp, yp = get_winsize(1)
    old_attrs = termios.tcgetattr(0)
    master, slave = os.openpty()
    set_winsize(master, max(rows - 1, 1), cols, xp, yp)
    termios.tcsetattr(slave, termios.TCSANOW, old_attrs)

    pid = os.fork()
    if pid == 0:
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        for fd in (0, 1, 2):
            os.dup2(slave, fd)
        for fd in (master, slave):
            if fd > 2:
                os.close(fd)
        os.execvp(shell, [shell])
    os.close(slave)

    wr, ww = os.pipe()
    for fd in (wr, ww):
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
    signal.signal(signal.SIGWINCH, lambda *_: os.write(ww, b"w"))

    tty.setraw(0)
    to_term, to_child = bytearray(), bytearray()
    row0, typed = cursor_row()
    to_child += typed
    if row0 is None or row0 >= rows:
        # the cursor sits where the strip will be: scroll everything up one line to
        # free the last row, then park the cursor on the new last body row
        os.write(1, b"\x1b[%d;1H\n\x1b[1;%dr\x1b[%d;1H" % (rows, max(rows - 1, 1), max(rows - 1, 1)))
    need_region, need_size = True, False
    last_child_read, next_paint = 0.0, 0.0
    key_ema = out_ema = 0.0
    key_n = out_n = 0
    t_rate = time.perf_counter()
    exit_code = 0

    try:
        while True:
            now = time.perf_counter()
            rl = [0, master, wr]
            wl = []
            if to_term:
                wl.append(1)
            if to_child:
                wl.append(master)
            timeout = max(0.0, min(next_paint - now, 0.05))
            r, w, _ = select.select(rl, wl, [], timeout)

            if wr in r:
                try:
                    os.read(wr, 64)
                except BlockingIOError:
                    pass
                need_size = True
            if 0 in r:
                data = os.read(0, 65536)
                if not data:
                    break
                key_n += len(data)
                to_child += data
            if master in r:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                out_n += len(data)
                last_child_read = now
                to_term += data
                if any(t in data for t in REGION_TRIGGERS):
                    need_region = True
            if 1 in w:
                n = os.write(1, bytes(to_term[:65536]))
                del to_term[:n]
            if master in w:
                n = os.write(master, bytes(to_child[:4096]))
                del to_child[:n]

            if need_size:
                need_size = False
                rows, cols, xp, yp = get_winsize(1)
                set_winsize(master, max(rows - 1, 1), cols, xp, yp)
                need_region = True

            now = time.perf_counter()
            if now - t_rate >= 0.25:
                span = now - t_rate
                # keystrokes arrive as ~1 byte each; smooth to a rate per second
                key_ema = 0.6 * key_ema + 0.4 * (key_n / span)
                out_ema = 0.6 * out_ema + 0.4 * (out_n / span)
                keys[0], outr[0] = key_ema, out_ema
                key_n = out_n = 0
                t_rate = now

            if now >= next_paint and not to_term and now - last_child_read > 0.003:
                if args.pattern:
                    ph = now * 1.5
                    levels[:] = np.clip(
                        (np.sin(np.arange(n_shown) * 0.21 - ph) * 2.0 + 1.6), 0, 3).astype(np.uint8)
                # show the strip only while the simulation is alive, so the shader
                # keeps its baked animation while the connectome loads (or if the
                # sim ever dies)
                live = args.pattern or (int(ready[0]) == 1 and sim is not None and sim.is_alive())
                to_term += paint_bytes(encode_cells(levels) if live else [], rows, cols, need_region)
                need_region = False
                next_paint = now + 1.0 / args.fps

            try:
                if os.waitpid(pid, os.WNOHANG)[0]:
                    break
            except ChildProcessError:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if to_term:
                os.write(1, bytes(to_term))
            os.write(1, b"\x1b[r\x1b7\x1b[%d;1H\x1b[0m\x1b[2K\x1b8" % rows)
        except OSError:
            pass
        termios.tcsetattr(0, termios.TCSADRAIN, old_attrs)
        if sim is not None:
            sim.terminate()
        try:
            os.kill(pid, signal.SIGHUP)
        except ProcessLookupError:
            pass
        try:
            _, status = os.waitpid(pid, 0)
            exit_code = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            pass
        shm.close()
        shm.unlink()
    return max(exit_code, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--speed", type=float, default=0.1,
                    help="simulated seconds per wall second (default 0.1; ~35%% of one core)")
    ap.add_argument("--fps", type=float, default=30.0, help="strip repaint rate")
    ap.add_argument("--shell", default=None, help="shell to host (default $SHELL)")
    ap.add_argument("--subset", default=str(HERE / "flybrain_subset.npz"))
    ap.add_argument("--rc", action="store_true",
                    help="use from ~/.zshrc: silently start a plain shell when already inside a bridge")
    ap.add_argument("--pattern", action="store_true",
                    help="paint a synthetic wave instead of running the simulation")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
