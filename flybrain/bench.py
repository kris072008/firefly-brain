"""Does the brain actually beat a dumb controller?

Runs the same arena, same seed, under four drivers and counts what each eats.
A fly flying in a straight line through a field of dots collects some by
accident, so "it ate 8" means nothing on its own.

    python -m flybrain.bench            # 60 s per driver, real connectome
    python -m flybrain.bench --dn DNp58 --seconds 120
"""
import random
import sys

import numpy as np

from .brain import Brain, Params
from .game import BrainController, World


def run(driver, seconds, seed, brain=None, ctrl=None, fps=20):
    world = World(seed)
    dt = 1.0 / fps
    rng = random.Random(seed)
    heading_bias = 0.0
    for _ in range(int(seconds * fps)):
        if driver == "brain":
            turn, _, _ = ctrl.tick(world.sense(), dt)
        elif driver == "brain-inverted":
            turn, _, _ = ctrl.tick(world.sense(), dt)
            turn = -turn
        elif driver == "random":
            heading_bias += rng.uniform(-1.0, 1.0)
            heading_bias *= 0.8
            turn = heading_bias
        else:                       # straight
            turn = 0.0
        world.advance(turn, dt)
    return world.fly.eaten, world.fly.poisoned


def main():
    argv = sys.argv[1:]
    dn = "DNg67"
    seconds = 60.0
    if "--dn" in argv:
        dn = argv[argv.index("--dn") + 1]
    if "--seconds" in argv:
        seconds = float(argv[argv.index("--seconds") + 1])

    from .data import load
    ann, W, _ = load()
    print(f"steering on {dn}, {seconds:.0f} s per driver, 3 seeds each\n")
    print(f"{'driver':<16}{'sugar':>8}{'bitter':>8}{'ratio':>9}")

    for driver in ("brain", "brain-inverted", "random", "straight"):
        tot_s = tot_b = 0
        for seed in (0, 1, 2):
            brain = ctrl = None
            if driver.startswith("brain"):
                brain = Brain(ann, W, Params())
                ctrl = BrainController(brain, cell_type=dn)
            s, b = run(driver, seconds, seed, brain, ctrl)
            tot_s += s
            tot_b += b
        ratio = tot_s / tot_b if tot_b else float("inf")
        print(f"{driver:<16}{tot_s:>8}{tot_b:>8}{ratio:>9.2f}")

    print("\nThe brain is doing something only if it beats BOTH random and")
    print("straight on the sugar-to-bitter ratio. If inverted scores the same,")
    print("the sign of the readout is not carrying information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
