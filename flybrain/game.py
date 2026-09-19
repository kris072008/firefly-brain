"""A real fly brain plays a game.

Sugar and bitter gustatory receptor neurons are driven by what is in front of
the fly; the two DNa02 descending neurons -- the fly's actual steering
command neurons -- are read out and turned into a turn rate. Nothing about
"seek food, avoid poison" is programmed. It comes out of the wiring.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

W_ARENA, H_ARENA = 820, 480
SENSE_RANGE = 230.0
MAX_RATE = 150.0           # Hz drive onto gustatory neurons, as in Shiu et al.
SPEED = 90.0               # px/s
TURN_GAIN = 6.0            # rad/s per unit of normalised DNa02 asymmetry


@dataclass
class Fly:
    x: float = W_ARENA / 2
    y: float = H_ARENA / 2
    heading: float = 0.9
    eaten: int = 0
    poisoned: int = 0


class World:
    def __init__(self, seed: int = 0, n_food: int = 16, n_poison: int = 8,
                 swap: bool = False):
        # swap=True feeds the green dots to the bitter receptors and the red
        # ones to the sugar receptors, so "avoid green" would have to come out
        # of the real bitter pathway rather than being written by hand.
        self.swap = swap
        self.rng = random.Random(seed)
        self.fly = Fly()
        self.food = [self._spot() for _ in range(n_food)]
        self.poison = [self._spot() for _ in range(n_poison)]

    def _spot(self):
        return [self.rng.uniform(40, W_ARENA - 40), self.rng.uniform(40, H_ARENA - 40)]

    def sense(self) -> dict[str, float]:
        """How strongly each antenna sees sugar and bitter, 0..1 per side."""
        out = {"sugar_left": 0.0, "sugar_right": 0.0,
               "bitter_left": 0.0, "bitter_right": 0.0}
        f = self.fly
        pairs = (("bitter", self.food), ("sugar", self.poison)) if self.swap \
            else (("sugar", self.food), ("bitter", self.poison))
        for kind, items in pairs:
            for sx, sy in items:
                dx = (sx - f.x + W_ARENA / 2) % W_ARENA - W_ARENA / 2
                dy = (sy - f.y + H_ARENA / 2) % H_ARENA - H_ARENA / 2
                d = math.hypot(dx, dy)
                if d > SENSE_RANGE:
                    continue
                rel = math.atan2(dy, dx) - f.heading
                rel = (rel + math.pi) % (2 * math.pi) - math.pi
                if abs(rel) > math.pi / 2:
                    continue
                # linear falloff, and take the strongest item rather than a
                # sum: squaring made even nearby food a fraction of MAX_RATE,
                # far too weak to push the network past threshold
                strength = (1 - d / SENSE_RANGE) * math.cos(rel)
                side = "left" if rel < 0 else "right"
                key = f"{kind}_{side}"
                out[key] = min(1.0, max(out[key], strength))
        return out

    def advance(self, turn: float, dt: float):
        f = self.fly
        f.heading = (f.heading + turn * dt) % (2 * math.pi)
        # wrap instead of clamping: a clamped fly parks against a wall facing
        # out, senses nothing, and the run is over
        f.x = (f.x + math.cos(f.heading) * SPEED * dt) % W_ARENA
        f.y = (f.y + math.sin(f.heading) * SPEED * dt) % H_ARENA
        for items, attr in ((self.food, "eaten"), (self.poison, "poisoned")):
            for s in items:
                ddx = (s[0] - f.x + W_ARENA / 2) % W_ARENA - W_ARENA / 2
                ddy = (s[1] - f.y + H_ARENA / 2) % H_ARENA - H_ARENA / 2
                if math.hypot(ddx, ddy) < 18:
                    setattr(f, attr, getattr(f, attr) + 1)
                    s[:] = self._spot()


class BrainController:
    """Binds world senses to sensory neurons and DNa02 to a turn rate."""

    #: Descending neuron used to steer. DNa02 is the fly's textbook steering
    #: command neuron, but it is driven by vision, not taste, and stays silent
    #: under gustatory input. DNge059 is what `flybrain calibrate` picks out:
    #: strong and near-symmetric to sugar (274 Hz left / 271 Hz right), zero to
    #: bitter. The symmetry matters -- a lopsided pair makes the fly circle.
    STEERING = "DNge059"

    def __init__(self, brain, dn_left=None, dn_right=None, cell_type=None,
                 invert=False):
        self.brain = brain
        a = brain.ann
        sugar = brain.select(cell_sub_class="sugar/water")
        bitter = brain.select(cell_sub_class="bitter")
        side = a["side"].values
        self.groups = {
            "sugar_left": sugar[side[sugar] == "left"],
            "sugar_right": sugar[side[sugar] == "right"],
            "bitter_left": bitter[side[bitter] == "left"],
            "bitter_right": bitter[side[bitter] == "right"],
        }
        if dn_left is None or dn_right is None:
            name = cell_type or self.STEERING
            dn = brain.select(cell_type=name)
            dn_left = dn[side[dn] == "left"]
            dn_right = dn[side[dn] == "right"]
            self.name = name
        else:
            self.name = cell_type or "custom"
        self.dn_left = np.asarray(dn_left, dtype=np.int32)
        self.dn_right = np.asarray(dn_right, dtype=np.int32)
        if not (self.dn_left.size and self.dn_right.size):
            raise RuntimeError("no steering neurons selected on one or both sides")
        # Which way a DNge059 asymmetry actually turns the fly is not
        # something the connectome tells us -- the wiring gives a left/right
        # difference, not a sign convention. `invert` flips it.
        self.invert = invert
        self.rng = np.random.default_rng(0)
        self._l = self._r = 0.0
        self.last_drive = {}
        self.rates = {}

    def tick(self, senses: dict[str, float], dt: float) -> tuple[float, float, float]:
        """Run the brain for dt seconds under `senses`; return (turn, L Hz, R Hz)."""
        # Gain control. Raw distance-scaled strength peaks around 0.3, which
        # is ~45 Hz -- far below the 150 Hz on 129 neurons that the feeding
        # experiment needed to drive anything downstream, so the brain sat
        # silent. Normalise so the strongest currently visible item always
        # drives its own population at MAX_RATE; direction is still carried
        # entirely by the left/right ratio, which is what the readout uses.
        # Sensory adaptation does something similar in a real fly.
        # Normalise sugar and bitter SEPARATELY. Sharing one peak meant a
        # nearby bitter dot at 69% squashed sugar from 14% down to 30 Hz, and
        # the sugar pathway never reached threshold.
        stim = {}
        self.rates = {}
        for kind in ("sugar", "bitter"):
            peak = max(senses.get(f"{kind}_left", 0.0),
                       senses.get(f"{kind}_right", 0.0))
            if peak <= 1e-3:
                continue
            for side in ("left", "right"):
                key = f"{kind}_{side}"
                rate = senses.get(key, 0.0) / peak * MAX_RATE
                self.rates[key] = rate
                if rate > 1.0:
                    for i in self.groups[key]:
                        stim[int(i)] = rate
        steps = max(1, int(round(dt / self.brain.p.dt)))
        nl = nr = 0
        for _ in range(steps):
            self.brain.step(stim, self.rng)
            m = self.brain.fired_mask
            nl += int(m[self.dn_left].sum())
            nr += int(m[self.dn_right].sum())
        window = steps * self.brain.p.dt
        # smooth the readout -- two neurons per side is a noisy signal
        self._l = 0.6 * self._l + 0.4 * (nl / self.dn_left.size / window)
        self._r = 0.6 * self._r + 0.4 * (nr / self.dn_right.size / window)
        total = self._l + self._r
        # DNa02 drives an ipsilateral turn: more right-side firing -> turn right
        asym = 0.0 if total < 1e-6 else (self._r - self._l) / max(total, 20.0)
        self.last_drive = senses
        turn = asym * TURN_GAIN
        return (-turn if self.invert else turn), self._l, self._r
