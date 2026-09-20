"""Leaky integrate-and-fire simulation of the whole FlyWire connectome.

Parameters follow Shiu et al. 2024 (Nature 634:210) -- the paper that showed
the wiring diagram alone predicts real fly behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Params:
    dt: float = 0.5e-3          # s, integration step
    tau_m: float = 20e-3        # membrane time constant
    tau_syn: float = 5e-3       # synaptic current decay
    v_rest: float = -52e-3
    v_reset: float = -52e-3
    v_thresh: float = -45e-3
    refractory: float = 2.2e-3
    epsp: float = 0.275e-3      # volts added per synapse per presynaptic spike
    delay: float = 1.8e-3       # axonal delay
    scale: float = 1.0          # global multiplier on synaptic strength


class Brain:
    """A whole fly brain you can step one timestep at a time."""

    def __init__(self, ann: pd.DataFrame, W: sp.csr_matrix, params: Params | None = None):
        self.ann = ann
        self.W = W.astype(np.float32)
        self.p = params or Params()
        self.n = W.shape[0]
        p = self.p
        self.v = np.full(self.n, p.v_rest, dtype=np.float32)
        self.i_syn = np.zeros(self.n, dtype=np.float32)
        self._tmp = np.empty(self.n, dtype=np.float32)
        self.refrac = np.zeros(self.n, dtype=np.float32)
        self._decay_i = np.float32(np.exp(-p.dt / p.tau_syn))
        self._dt_over_tau = np.float32(p.dt / p.tau_m)
        self._gain = np.float32(p.epsp * p.scale)
        d = max(1, int(round(p.delay / p.dt)))
        self._buf = [np.empty(0, dtype=np.int32) for _ in range(d)]
        self._cursor = 0
        self.t = 0.0
        self.spike_counts = np.zeros(self.n, dtype=np.int64)
        self.fired_mask = np.zeros(self.n, dtype=bool)

    def reset(self) -> None:
        """Back to the resting state: no voltage, current, refractoriness or spikes in flight."""
        self.v[:] = self.p.v_rest
        self.i_syn[:] = 0
        self.refrac[:] = 0
        self._buf = [np.empty(0, dtype=np.int32) for _ in self._buf]

    # ---- selecting neurons -------------------------------------------------
    def select(self, **kwargs) -> np.ndarray:
        """select(cell_sub_class="sugar/water", side="left") -> row indices."""
        mask = np.ones(len(self.ann), dtype=bool)
        for col, want in kwargs.items():
            vals = want if isinstance(want, (list, tuple, set)) else [want]
            mask &= self.ann[col].isin(list(vals)).values
        return np.flatnonzero(mask).astype(np.int32)

    # ---- simulation --------------------------------------------------------
    def step(self, stim: dict[int, float] | tuple[np.ndarray, np.ndarray] | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
        """Advance dt. `stim` maps neuron index -> Poisson drive rate in Hz,
        or is a pair of arrays (indices, rates) for large drive sets.

        Returns the indices of neurons that spiked this step.
        """
        p = self.p
        rng = rng or np.random.default_rng()

        # 1. deliver spikes that left their soma `delay` ago
        arriving = self._buf[self._cursor]
        if arriving.size:
            # row-slice the CSR: only presynaptic neurons that fired contribute
            self.i_syn += np.asarray(self.W[arriving].sum(axis=0)).ravel() * self._gain

        # 2. integrate  dv/dt = (-(v - v_rest) + g) / tau_m,  dg/dt = -g / tau_syn
        #    A synapse's `epsp` is the jump in g, not the jump in v: with these
        #    time constants the resulting peak deflection in v is about 0.16 of
        #    it, so one synapse is worth ~0.043 mV and roughly 160 coincident
        #    synapses are needed to reach threshold. Adding epsp straight to v
        #    makes every neuron ~23x too excitable and the brain saturates.
        #    Done in place with `where=` masks over the whole array: identical
        #    arithmetic to indexing with the mask, without the temporaries.
        live = self.refrac <= 0
        tmp = self._tmp
        np.subtract(p.v_rest, self.v, out=tmp)
        tmp += self.i_syn
        tmp *= self._dt_over_tau
        np.add(self.v, tmp, out=self.v, where=live)
        self.i_syn *= self._decay_i
        np.subtract(self.refrac, p.dt, out=self.refrac, where=~live)

        # 3. external drive: independent Poisson spikes forced onto sensory cells
        forced = np.empty(0, dtype=np.int32)
        if isinstance(stim, tuple):
            idx, rate = stim
            forced = idx[rng.random(idx.size) < rate * p.dt]
        elif stim:
            idx = np.fromiter(stim.keys(), dtype=np.int32, count=len(stim))
            rate = np.fromiter(stim.values(), dtype=np.float32, count=len(stim))
            forced = idx[rng.random(idx.size) < rate * p.dt]

        # 4. threshold
        fired = np.flatnonzero(self.v >= p.v_thresh).astype(np.int32)
        if forced.size:
            fired = np.union1d(fired, forced).astype(np.int32)
        self.fired_mask[:] = False
        if fired.size:
            self.fired_mask[fired] = True
            self.v[fired] = p.v_reset
            self.refrac[fired] = p.refractory
            self.spike_counts[fired] += 1

        # 5. queue them for delayed delivery
        self._buf[self._cursor] = fired
        self._cursor = (self._cursor + 1) % len(self._buf)
        self.t += p.dt
        return fired

    def run(self, seconds: float, stim=None, rng=None) -> np.ndarray:
        """Run for `seconds` and return firing rates in Hz for every neuron."""
        start = self.spike_counts.copy()
        steps = int(round(seconds / self.p.dt))
        for _ in range(steps):
            self.step(stim, rng)
        return (self.spike_counts - start) / seconds
