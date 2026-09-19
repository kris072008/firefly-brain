"""python -m flybrain [check|synthetic|feeding|lateral|calibrate|play|columns] [--scale N] [--dn CELLTYPE] [--invert] [--swap]"""
import sys
import time

import numpy as np


def _load(cmd):
    if cmd == "synthetic":
        from .synthetic import make
        return make()
    from .data import load
    return load()


def _check():
    """Verify the integrator against the analytic EPSP for a single synapse."""
    import pandas as pd
    import scipy.sparse as sp
    from .brain import Brain, Params

    tau_m, tau_syn = 20e-3, 5e-3
    r = tau_syn / (tau_m - tau_syn)
    predicted = 0.275 * r * ((tau_syn / tau_m) ** r
                             - (tau_syn / tau_m) ** (tau_m / (tau_m - tau_syn)))

    ann = pd.DataFrame({"root_id": [1, 2], "top_nt": ["acetylcholine"] * 2,
                        "side": ["left", "right"], "cell_type": [None, None],
                        "cell_class": [None, None], "super_class": [None, None],
                        "cell_sub_class": [None, None]})
    W = sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=np.float32))
    b = Brain(ann, W, Params(dt=0.1e-3))
    b.v[0] = b.p.v_thresh + 1e-3
    peak = -1.0
    for _ in range(800):
        b.step()
        peak = max(peak, float(b.v[1] - b.p.v_rest))
    measured = peak * 1e3
    need = (b.p.v_thresh - b.p.v_rest) / peak
    print(f"one synapse produces {measured:.4f} mV   (analytic {predicted:.4f} mV)")
    print(f"synapses needed to reach threshold: {need:.0f}")
    ok = abs(measured - predicted) < 0.002
    print("PASS" if ok else "FAIL: integrator does not match the model equations")
    return 0 if ok else 1


def main():
    argv = sys.argv[1:]
    scale = 1.0
    invert = "--invert" in argv
    if invert:
        argv.remove("--invert")
    swap = "--swap" in argv
    if swap:
        argv.remove("--swap")
    dn = None
    if "--dn" in argv:
        i = argv.index("--dn")
        dn = argv[i + 1]
        del argv[i:i + 2]
    if "--scale" in argv:
        i = argv.index("--scale")
        scale = float(argv[i + 1])
        del argv[i:i + 2]
    cmd = argv[0] if argv else "play"

    if cmd == "check":
        return _check()
    if cmd == "columns":
        # debug helper: what does the downloaded connection table look like?
        import pandas as pd
        from .data import download
        _, con_path = download()
        con = pd.read_feather(con_path)
        print(f"{len(con):,} rows")
        print(con.dtypes)
        print(con.head())
        return 0

    if cmd not in {"synthetic", "feeding", "calibrate", "lateral", "play"}:
        print(__doc__)
        return 1

    ann, W, _ = _load(cmd)
    from .brain import Brain, Params
    from .game import BrainController, World
    params = Params(scale=scale)

    if cmd == "synthetic":
        # engine smoke test on random wiring: checks speed and plumbing only.
        # epsp is scaled down because a randomly wired net is not balanced and
        # would otherwise seize. Nothing here means anything biologically.
        brain = Brain(ann, W, Params(epsp=0.275e-3 / 40, scale=scale))
        ctrl = BrainController(brain, cell_type=dn)
        world = World(0)
        print(f"{brain.n:,} neurons, {W.nnz:,} connections")
        print("sensory groups:", {k: len(v) for k, v in ctrl.groups.items()})
        print(f"steering neurons: {ctrl.dn_left.size} left, {ctrl.dn_right.size} right")
        t0 = time.time()
        for _ in range(20):
            turn, l, r = ctrl.tick(world.sense(), 1 / 20)
            world.advance(turn, 1 / 20)
        el = time.time() - t0
        print(f"1.0 s of whole-brain simulation in {el:.2f} s "
              f"({1/el:.1f}x real time)")
        print(f"neurons that fired: {(brain.spike_counts > 0).sum():,}, "
              f"spikes: {brain.spike_counts.sum():,}")
        print("\nengine works. now run:  python -m flybrain feeding")
        return 0

    brain = Brain(ann, W, params)

    if cmd == "feeding":
        # The published check: driving sugar-sensing taste neurons should make
        # the ingestion motor neurons fire, and bitter should not.
        # Shiu et al. 2024, Nature 634:210.
        ing = brain.select(cell_sub_class="ingestion_motor_neuron")
        print(f"{'condition':<16}{'ingestion MNs (Hz)':>20}{'neurons > 1 Hz':>16}")
        for label, sub in (("sugar", "sugar/water"), ("bitter", "bitter"),
                           ("sugar+bitter", None)):
            b2 = Brain(ann, W, params)
            if sub is None:
                drive = np.union1d(b2.select(cell_sub_class="sugar/water"),
                                   b2.select(cell_sub_class="bitter"))
            else:
                drive = b2.select(cell_sub_class=sub)
            rates = b2.run(1.0, {int(i): 150.0 for i in drive},
                           np.random.default_rng(0))
            print(f"{label:<16}{rates[ing].mean():>20.1f}{int((rates > 1).sum()):>16,}")
            if label == "sugar":
                sugar_rates = rates

        ceiling = 1.0 / (brain.p.refractory + brain.p.dt)
        active = sugar_rates[sugar_rates > 1]
        n_active = active.size
        if n_active > 5000 or (n_active and active.mean() > 0.5 * ceiling):
            print(f"\nWARNING: {n_active:,} neurons active at a mean of "
                  f"{active.mean():.0f} Hz against a {ceiling:.0f} Hz refractory "
                  "ceiling -- the network is saturating, not computing. "
                  "Try --scale 0.5")
        else:
            print("\ntop responders to sugar:")
            for i in np.argsort(sugar_rates)[::-1][:12]:
                a = ann.iloc[i]
                name = a.cell_type if isinstance(a.cell_type, str) else a.cell_class
                print(f"  {sugar_rates[i]:7.1f} Hz  {name}  "
                      f"({a.super_class}, {a.side})")
        return 0

    if cmd == "lateral":
        # The game drives ONE side of the sugar neurons, not all 129. Does a
        # one-sided stimulus reach the descending neurons at all, and does any
        # of them tell left from right? That is what steering needs.
        dn = brain.select(super_class="descending")
        sides = brain.ann["side"].values
        out = {}
        for label in ("left", "right", "both"):
            b2 = Brain(ann, W, params)
            sugar = b2.select(cell_sub_class="sugar/water")
            drive = sugar if label == "both" else sugar[sides[sugar] == label]
            rates = b2.run(1.0, {int(i): 150.0 for i in drive},
                           np.random.default_rng(0))
            out[label] = rates
            print(f"stimulus {label:<6} {drive.size:>3} neurons  ->  "
                  f"{int((rates > 1).sum()):>5,} neurons active, "
                  f"{int((rates[dn] > 1).sum()):>3} descending")

        if int((out["left"][dn] > 1).sum()) == 0 and int((out["right"][dn] > 1).sum()) == 0:
            print("\nNo descending neuron responds to a one-sided sugar stimulus.")
            print("The game cannot steer on taste alone -- this is a property of")
            print("the connectome, not a bug in the simulation.")
            return 0

        # rank descending neurons by how well they separate left from right
        print(f"\n{'cell_type':<14}{'side':<7}{'L-stim Hz':>11}{'R-stim Hz':>11}"
              f"{'difference':>12}")
        diff = out["left"][dn] - out["right"][dn]
        for k in np.argsort(np.abs(diff))[::-1][:15]:
            if abs(diff[k]) < 1:
                break
            a = ann.iloc[dn[k]]
            print(f"{str(a.cell_type):<14}{str(a.side):<7}"
                  f"{out['left'][dn][k]:>11.1f}{out['right'][dn][k]:>11.1f}"
                  f"{diff[k]:>+12.1f}")
        print("\nA usable steering pair is one cell_type with a left and a right")
        print("copy whose differences have opposite signs. Pass it with --dn.")
        return 0

    if cmd == "calibrate":
        dn = brain.select(super_class="descending")
        out = {}
        for name, sub in (("sugar", "sugar/water"), ("bitter", "bitter")):
            b2 = Brain(ann, W, params)
            drive = b2.select(cell_sub_class=sub)
            out[name] = b2.run(1.0, {int(i): 150.0 for i in drive},
                               np.random.default_rng(0))[dn]
        order = np.argsort(out["sugar"] - out["bitter"])[::-1][:20]
        print(f"{'cell_type':<16}{'side':<7}{'sugar Hz':>10}{'bitter Hz':>11}")
        for k in order:
            a = ann.iloc[dn[k]]
            print(f"{str(a.cell_type):<16}{str(a.side):<7}"
                  f"{out['sugar'][k]:>10.1f}{out['bitter'][k]:>11.1f}")
        print("\nPick the best left/right pair as your steering readout.")
        return 0

    from .render import play
    play(brain, cell_type=dn, invert=invert, swap=swap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
