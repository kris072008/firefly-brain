"""Does a ONE-SIDED sugar stimulus reach the descending neurons?

The game drives only the left or only the right sugar receptors, never all
129 at once the way the feeding experiment does. This asks whether that is
enough to produce any descending activity, and whether any descending neuron
tells left from right -- which is what steering needs.

    python -m flybrain.lateral            # real connectome
    python -m flybrain.lateral --synthetic  # random wiring, engine test only
"""
import sys

import numpy as np

from .brain import Brain, Params


def main():
    if "--synthetic" in sys.argv:
        from .synthetic import make
        ann, W, _ = make()
        params = Params(epsp=0.275e-3 / 40)
    else:
        from .data import load
        ann, W, _ = load()
        params = Params()

    probe = Brain(ann, W, params)
    dn = probe.select(super_class="descending")
    sugar = probe.select(cell_sub_class="sugar/water")
    sides = ann["side"].values

    out = {}
    for label in ("left", "right", "both"):
        b = Brain(ann, W, params)
        drive = sugar if label == "both" else sugar[sides[sugar] == label]
        rates = b.run(1.0, {int(i): 150.0 for i in drive},
                      np.random.default_rng(0))
        out[label] = rates
        print(f"stimulus {label:<6} {drive.size:>3} neurons  ->  "
              f"{int((rates > 1).sum()):>6,} neurons active, "
              f"{int((rates[dn] > 1).sum()):>3} descending")

    if (out["left"][dn] > 1).sum() == 0 and (out["right"][dn] > 1).sum() == 0:
        print("\nNo descending neuron responds to a one-sided sugar stimulus.")
        print("The game cannot steer on taste alone. That is a property of the")
        print("connectome, not a bug in the simulation.")
        return 0

    diff = out["left"][dn] - out["right"][dn]
    print(f"\n{'cell_type':<14}{'side':<7}{'L-stim Hz':>11}{'R-stim Hz':>11}"
          f"{'difference':>12}")
    shown = 0
    for k in np.argsort(np.abs(diff))[::-1]:
        if abs(diff[k]) < 1 or shown >= 20:
            break
        a = ann.iloc[dn[k]]
        print(f"{str(a.cell_type):<14}{str(a.side):<7}"
              f"{out['left'][dn][k]:>11.1f}{out['right'][dn][k]:>11.1f}"
              f"{diff[k]:>+12.1f}")
        shown += 1
    if not shown:
        print("(no descending neuron separates left from right by even 1 Hz)")
    else:
        print("\nA usable steering pair is one cell_type appearing twice, with")
        print("left and right copies whose differences have opposite signs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
