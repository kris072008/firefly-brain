"""Bake a slice of the real FlyWire connectome into a Ghostty GLSL shader.

    python ghostty/bake.py                 # writes ghostty/flybrain.glsl
    python ghostty/bake.py -n 400 --slow 14

What it does
1. Runs the repo's LIF simulation on the full 139k-neuron brain under four
   sensory "scenes" (taste, sight, smell, touch) and records, for every neuron,
   its first-spike latency and spike count.
2. Picks ~N well-connected neurons that actually fire, spread over the frontal
   (x, y) projection so the brain's silhouette shows.
3. Keeps their strongest mutual synapses as edges.
4. Builds a coarse spatial grid (CSR lists of neurons/edges per cell) so the
   fragment shader only touches the handful of items near each pixel.
5. Emits everything as GLSL constant arrays into flybrain.glsl.

The shader replays the recorded latencies slowed down by --slow, so the
cascades you see are the model's real spike timing, not an invented animation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from scipy.spatial import cKDTree  # noqa: E402

from flybrain.brain import Brain  # noqa: E402
from flybrain.data import load  # noqa: E402

# name, selector on the annotation table, Poisson drive in Hz
SCENES = [
    ("taste", dict(cell_sub_class="sugar/water"), 150.0),
    ("sight", dict(cell_class="visual"), 150.0),
    ("smell", dict(cell_class="olfactory"), 100.0),
    ("touch", dict(cell_class="mechanosensory"), 100.0),
]
SIM_MS = 300.0          # simulated milliseconds per scene
KIND = {"acetylcholine": 0, "gaba": 1, "glutamate": 2,
        "dopamine": 3, "serotonin": 3, "octopamine": 3}

# Phase 2: typing. A grid of HX x HY cells covers the window; the cell under the
# cursor picks one baked cascade. The window is mapped onto the brain's bounding
# box (scaled by HMAP), and each cell force-fires the real neurons nearest to it.
HX, HY, HMAP = 10, 6, 0.92
TYP_SLOW = 8.0          # display slow-down of the typing cascades
TYP_SIM_MS = 200.0
TYP_TARGET = 30         # escalate the stimulus until this many shown neurons fire
# (neurons forced, Hz, burst ms) -- weakest first
LADDER = [(12, 200, 10), (40, 250, 20), (80, 300, 20), (160, 400, 30),
          (320, 500, 30), (640, 500, 40)]

# geometry, in units where the brain is 1.0 wide
CELL = 0.04
DENS_STEP = 0.02       # resolution of the neuropil haze grid
R_NEURON = 0.040        # max radius a neuron's glow can reach
R_EDGE = 0.022          # max radius a line / pulse can reach


def run_scenes(ann, W):
    n = W.shape[0]
    steps = int(SIM_MS / 0.5)
    first = np.full((len(SCENES), n), -1.0, dtype=np.float32)
    count = np.zeros((len(SCENES), n), dtype=np.float32)
    stimulated = np.zeros(n, dtype=bool)
    for m, (name, sel, rate) in enumerate(SCENES):
        b = Brain(ann, W)
        idx = b.select(**sel)
        stimulated[idx] = True
        stim = {int(i): rate for i in idx}
        rng = np.random.default_rng(m)
        for s in range(steps):
            fired = b.step(stim, rng)
            fresh = fired[first[m, fired] < 0]
            first[m, fresh] = s * 0.5
            count[m, fired] += 1
        print(f"  scene {name:<6} drive {len(idx):>5} neurons @ {rate:.0f} Hz -> "
              f"{int((first[m] >= 0).sum()):>6,} neurons fired")
    return first, count, stimulated


def select_subset(ann, W, first, stimulated, n_target, n_sensory):
    n = W.shape[0]
    A = abs(W)
    deg = np.asarray(A.sum(axis=1)).ravel() + np.asarray(A.sum(axis=0)).ravel()
    nsc = (first >= 0).sum(axis=0)
    x = ann.pos_x.values.astype(np.float64)
    y = ann.pos_y.values.astype(np.float64)

    # taste recruits far fewer neurons than the other scenes; favour them so
    # that wave is not empty on screen
    rare = np.zeros(n)
    for m in range(first.shape[0]):
        act_m = first[m] >= 0
        rare += act_m * (2.0 / max(np.log10(act_m.sum()), 1.0) - 0.4)
    score = np.log1p(deg) + 0.6 * nsc + 1.5 * rare
    active = nsc > 0

    # a few stimulated sensory neurons, so the waves visibly start somewhere
    sens = np.flatnonzero(stimulated & active & (ann.super_class.values == "sensory"))
    sens = sens[np.argsort(-deg[sens])][: n_sensory * 4]
    rng = np.random.default_rng(1)
    sens = rng.choice(sens, size=min(n_sensory, sens.size), replace=False)

    pool = np.flatnonzero(active & ~np.isin(np.arange(n), sens)
                          & (ann.super_class.values != "sensory"))
    # stratify over a grid so both hemispheres and the optic lobes are covered
    gx, gy = 22, 14
    xb = np.clip(((x[pool] - x[pool].min()) / np.ptp(x[pool]) * gx).astype(int), 0, gx - 1)
    yb = np.clip(((y[pool] - y[pool].min()) / np.ptp(y[pool]) * gy).astype(int), 0, gy - 1)
    cell = xb * gy + yb
    order = np.lexsort((-score[pool], cell))
    pool, cell = pool[order], cell[order]
    rank = np.zeros(len(pool), dtype=int)
    starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(pool)]):
        rank[s:e] = np.arange(e - s)
    take = np.lexsort((-score[pool], rank))[: n_target - len(sens)]
    sel = np.concatenate([sens, pool[take]])
    return sel, deg


def build_edges(W, sel, xy, max_edges, max_len, per_source):
    sub = W[sel][:, sel].toarray()
    np.fill_diagonal(sub, 0)
    cand = []
    for a in range(len(sel)):
        row = sub[a]
        for b in np.argsort(-np.abs(row))[:per_source * 3]:
            w = row[b]
            if abs(w) < 3:
                break
            ln = float(np.hypot(*(xy[a] - xy[b])))
            if ln > max_len or ln < 0.01:
                continue
            cand.append((abs(w) / (1.0 + ln / 0.12), a, b, w))
    cand.sort(reverse=True)
    out_count = np.zeros(len(sel), dtype=int)
    edges = []
    for _, a, b, w in cand:
        if out_count[a] >= per_source:
            continue
        out_count[a] += 1
        edges.append((a, b, abs(w)))
        if len(edges) >= max_edges:
            break
    return edges


def seg_hits_rect(a, b, lo, hi):
    """Liang-Barsky: does segment a-b touch the rect [lo, hi]?"""
    d = b - a
    t0, t1 = 0.0, 1.0
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if a[i] < lo[i] or a[i] > hi[i]:
                return False
        else:
            ta, tb = (lo[i] - a[i]) / d[i], (hi[i] - a[i]) / d[i]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                return False
    return True


def build_grid(xy, edges):
    pad = max(R_NEURON, R_EDGE) + 0.01
    gmin = xy.min(axis=0) - pad
    gmax = xy.max(axis=0) + pad
    gx = int(np.ceil((gmax[0] - gmin[0]) / CELL))
    gy = int(np.ceil((gmax[1] - gmin[1]) / CELL))
    n_start, n_list, e_start, e_list = [0], [], [0], []
    for cy in range(gy):
        for cx in range(gx):
            lo = gmin + np.array([cx, cy]) * CELL
            hi = lo + CELL
            nearest = np.clip(xy, lo, hi)
            near = np.flatnonzero(np.hypot(*(xy - nearest).T) <= R_NEURON)
            n_list.extend(near.tolist())
            n_start.append(len(n_list))
            elo, ehi = lo - R_EDGE, hi + R_EDGE
            for k, (a, b, _) in enumerate(edges):
                if seg_hits_rect(xy[a], xy[b], elo, ehi):
                    e_list.append(k)
            e_start.append(len(e_list))
    return dict(gmin=gmin, gx=gx, gy=gy, n_start=n_start, n_list=n_list,
                e_start=e_start, e_list=e_list)


def build_density(xy_all, aspect):
    """Blurred log-density of all neurons: the faint 'neuropil' haze."""
    from scipy.ndimage import gaussian_filter
    hx, hy = 0.56, aspect / 2 + 0.08
    nx, ny = int(round(2 * hx / DENS_STEP)), int(round(2 * hy / DENS_STEP))
    h, _, _ = np.histogram2d(xy_all[:, 0], xy_all[:, 1], bins=[nx, ny],
                             range=[[-hx, hx], [-hy, hy]])
    h = gaussian_filter(h, 1.2)
    h = np.sqrt(h / np.percentile(h[h > 0], 97))
    h = np.clip(h, 0, 1)
    h[h < 0.12] = 0.0
    return dict(hx=hx, hy=hy, nx=nx, ny=ny, vals=h.T.ravel())   # row-major, y down


def build_typing(ann, W, sel, xy_all, aspect):
    """Per grid cell: first-spike latency of each shown neuron after a burst of
    forced spikes in the ~M real neurons nearest that cell (real LIF sim)."""
    tree = cKDTree(xy_all)
    shown_tree = cKDTree(xy_all[sel])
    n = W.shape[0]
    steps = int(TYP_SIM_MS / 0.5)
    table = np.full((HX * HY, len(sel)), 255, dtype=np.uint8)
    rung_used = np.zeros((HY, HX), dtype=int)
    reach = np.zeros((HY, HX), dtype=int)
    for iy in range(HY):
        for ix in range(HX):
            p = np.array([((ix + .5) / HX - .5) * HMAP, ((iy + .5) / HY - .5) * aspect * HMAP])
            near_shown = sel[shown_tree.query(p, 3)[1]]
            for r, (m, rate, burst) in enumerate(LADDER):
                grp = np.union1d(tree.query(p, m)[1], near_shown)
                b = Brain(ann, W)
                first = np.full(n, -1.0)
                stim = {int(i): float(rate) for i in grp}
                rng = np.random.default_rng(iy * HX + ix)
                for s in range(steps):
                    fired = b.step(stim if s < burst * 2 else None, rng)
                    fresh = fired[first[fired] < 0]
                    first[fresh] = s * 0.5
                got = int((first[sel] >= 0).sum())
                if got >= TYP_TARGET:
                    break
            rung_used[iy, ix], reach[iy, ix] = r, got
            t = first[sel]
            q = np.round(t / 1000.0 * TYP_SLOW * 100.0)
            table[iy * HX + ix] = np.where((t >= 0) & (q < 255), q, 255).astype(np.uint8)
    return table, rung_used, reach


def pack_u8(flat):
    flat = np.concatenate([flat, np.full((-len(flat)) % 4, 255, dtype=np.uint8)]).astype(np.uint64)
    w = flat.reshape(-1, 4)
    return (w[:, 0] | (w[:, 1] << 8) | (w[:, 2] << 16) | (w[:, 3] << 24)).tolist()


def arr(vals, fmt, per_line=8):
    items = [fmt(v) for v in vals]
    lines = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return ",\n    ".join(lines)


def emit(path_tmpl, path_out, xy, aspect, kind, depth, wave, edges, ew, grid,
         slow, stagger, wave_dur, dens, typing):
    nn, ne, nw = len(xy), len(edges), wave.shape[0]
    parts = [
        f"const int NN = {nn};\nconst int NE = {ne};\nconst int NW = {nw};",
        f"const int GX = {grid['gx']};\nconst int GY = {grid['gy']};",
        f"const float CELL = {CELL};",
        f"const vec2 GMIN = vec2({grid['gmin'][0]:.4f}, {grid['gmin'][1]:.4f});",
        f"const float ASPECT = {aspect:.5f};",
        f"const int HX = {HX};\nconst int HY = {HY};\nconst float HMAP = {HMAP};"
        f"\nconst float TYP_DUR = {typing['dur']:.2f};",
        f"const uint TYP[{len(typing['packed'])}] = uint[](\n    "
        + arr(typing["packed"], lambda v: f"{v}u", 10) + "\n);",
        f"const int DX = {dens['nx']};\nconst int DY = {dens['ny']};\n"
        f"const float DSTEP = {DENS_STEP};\nconst vec2 DMIN = vec2({-dens['hx']:.4f}, {-dens['hy']:.4f});",
        f"const float DENS[DX * DY] = float[DX * DY](\n    "
        + arr(dens["vals"], lambda v: f"{v:.2f}".replace("0.", ".") if v else "0.", 24) + "\n);",
        f"const float STAGGER = {stagger:.3f};\nconst float LOOP = {stagger * nw:.3f};"
        f"\nconst float WAVE_DUR = {wave_dur:.3f};",
        f"const vec2 NPOS[NN] = vec2[NN](\n    "
        + arr(xy, lambda p: f"vec2({p[0]:.4f},{p[1]:.4f})", 4) + "\n);",
        f"const int NKIND[NN] = int[NN](\n    " + arr(kind, lambda v: str(int(v)), 32) + "\n);",
        f"const float NDEPTH[NN] = float[NN](\n    " + arr(depth, lambda v: f"{v:.2f}", 16) + "\n);",
        f"const ivec2 EDGE[NE] = ivec2[NE](\n    "
        + arr(edges, lambda e: f"ivec2({e[0]},{e[1]})", 8) + "\n);",
        f"const float EW[NE] = float[NE](\n    " + arr(ew, lambda v: f"{v:.2f}", 16) + "\n);",
        f"const int CELL_N_START[{len(grid['n_start'])}] = int[](\n    "
        + arr(grid["n_start"], str, 24) + "\n);",
        f"const int CELL_N[{len(grid['n_list'])}] = int[](\n    "
        + arr(grid["n_list"], str, 32) + "\n);",
        f"const int CELL_E_START[{len(grid['e_start'])}] = int[](\n    "
        + arr(grid["e_start"], str, 24) + "\n);",
        f"const int CELL_E[{len(grid['e_list'])}] = int[](\n    "
        + arr(grid["e_list"], str, 32) + "\n);",
        # x = first-spike time (s, slowed), y = period between spikes, z = spike count
        f"const vec3 WAVE[NW * NN] = vec3[NW * NN](\n    "
        + arr(wave.reshape(-1, 3), lambda w: f"vec3({w[0]:.3f},{w[1]:.3f},{w[2]:.0f})", 4) + "\n);",
    ]
    text = Path(path_tmpl).read_text().replace("//@@DATA@@", "\n\n".join(parts))
    Path(path_out).write_text(text)
    return len(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=400, help="neurons to keep")
    ap.add_argument("--sensory", type=int, default=48, help="of which stimulated sensory")
    ap.add_argument("--edges", type=int, default=520)
    ap.add_argument("--max-len", type=float, default=0.20, help="longest edge, brain widths")
    ap.add_argument("--per-source", type=int, default=3)
    ap.add_argument("--slow", type=float, default=14.0, help="display slow-down of the sim")
    ap.add_argument("--stagger", type=float, default=3.2)
    ap.add_argument("--out", default=str(HERE / "flybrain.glsl"))
    args = ap.parse_args()

    print("loading connectome ...")
    ann, W, _ = load()
    print("running real LIF simulation scenes ...")
    first, count, stimulated = run_scenes(ann, W)

    sel, deg = select_subset(ann, W, first, stimulated, args.n, args.sensory)
    print(f"selected {len(sel)} neurons")

    # frontal projection, y down (matches Ghostty's fragCoord on Metal). The
    # FlyWire frame is tilted, so rotate the principal axis onto the x axis.
    X = ann.pos_x.values.astype(np.float64)
    Y = ann.pos_y.values.astype(np.float64)
    c = np.cov(np.stack([X, Y]))
    theta = 0.5 * np.arctan2(2 * c[0, 1], c[0, 0] - c[1, 1])
    ct, st = np.cos(-theta), np.sin(-theta)
    RX, RY = ct * X - st * Y, st * X + ct * Y
    lo, hi = np.percentile(RX, 0.3), np.percentile(RX, 99.7)
    width = hi - lo
    cx = (lo + hi) / 2
    cy = (np.percentile(RY, 0.3) + np.percentile(RY, 99.7)) / 2
    xy_all = np.stack([(RX - cx) / width, (RY - cy) / width], axis=1)
    xy = xy_all[sel]
    z = ann.pos_z.values[sel].astype(np.float64)
    aspect = float((np.percentile(xy_all[:, 1], 99.7) - np.percentile(xy_all[:, 1], 0.3)))
    print(f"levelled by {np.degrees(theta):.1f} deg; brain aspect {aspect:.3f}")
    depth = (z - z.min()) / max(np.ptp(z), 1)
    dens = build_density(xy_all, aspect)

    nt = ann.top_nt.fillna("acetylcholine").str.lower().values[sel]
    kind = np.array([KIND.get(t, 0) for t in nt])

    edges = build_edges(W, sel, xy, args.edges, args.max_len, args.per_source)
    ew = np.array([e[2] for e in edges])
    ew = np.sqrt(ew / ew.max())
    print(f"kept {len(edges)} edges (strongest synapse counts "
          f"{int(max(e[2] for e in edges))} .. {int(min(e[2] for e in edges))})")

    grid = build_grid(xy, edges)
    ncell = grid["gx"] * grid["gy"]
    nl = np.diff(grid["n_start"])
    el = np.diff(grid["e_start"])
    print(f"grid {grid['gx']}x{grid['gy']} = {ncell} cells; neurons/cell "
          f"avg {nl.mean():.1f} max {nl.max()}; edges/cell avg {el.mean():.1f} max {el.max()}")

    # wave table from the real simulation, slowed down for display
    nw = len(SCENES)
    wave = np.zeros((nw, len(sel), 3), dtype=np.float64)
    for m in range(nw):
        t0 = first[m, sel] / 1000.0 * args.slow
        c = count[m, sel]
        rate_hz = c / (SIM_MS / 1000.0)
        # spikes are spaced by the mean inter-spike interval, floored so the
        # display never strobes faster than ~14 Hz
        period = np.where(c > 1, np.maximum(args.slow / np.maximum(rate_hz, 1e-3), 0.07), 0.0)
        cnt = np.minimum(c, 60)
        wave[m, :, 0] = np.where(first[m, sel] >= 0, t0, -1.0)
        wave[m, :, 1] = period
        wave[m, :, 2] = cnt
        fired = int((first[m, sel] >= 0).sum())
        print(f"  wave {SCENES[m][0]:<6} lights {fired}/{len(sel)} shown neurons")
    tail = wave[:, :, 0] + np.maximum(wave[:, :, 2] - 1, 0) * wave[:, :, 1]
    wave_dur = float(tail.max() + 0.6)
    print(f"wave duration {wave_dur:.2f}s, loop {args.stagger * nw:.1f}s")

    print("typing cascades (real sim, per window cell) ...")
    table, rung, reach = build_typing(ann, W, sel, xy_all, aspect)
    print(f"  shown neurons reached per cell: min {reach.min()} median "
          f"{int(np.median(reach))} max {reach.max()}; stimulus rung used "
          f"{np.bincount(rung.ravel(), minlength=len(LADDER)).tolist()}")
    latest = table[table < 255].max() / 100.0
    typing = dict(packed=pack_u8(table.ravel()), dur=float(latest + 0.7))

    size = emit(HERE / "flybrain.template.glsl", args.out, xy, aspect, kind, depth, wave,
                edges, ew, grid, args.slow, args.stagger, wave_dur, dens, typing)
    print(f"wrote {args.out} ({size / 1024:.0f} KiB)")

    np.savez(HERE / "flybrain_subset.npz", typing_table=table, sel=sel, xy=xy, kind=kind,
             root_id=ann.root_id.values[sel], edges=np.array(edges),
             scenes=np.array([s[0] for s in SCENES]))


if __name__ == "__main__":
    main()
