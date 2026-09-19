# A real fly brain plays a game

The full adult *Drosophila* connectome — 139,248 neurons and ~2.7 million
synaptic connections, mapped by the FlyWire consortium and released free —
runs here as a spiking network, and its own steering neurons drive a fly
around an arena looking for sugar.

Nothing in the game logic says "go towards food". The behaviour, if it
appears, comes out of the wiring diagram.

## Run it

```bash
pip install -r requirements.txt
python -m flybrain.server       # 3D chess vs the connectome in a browser

python -m flybrain check       # verify the integrator against the analytic EPSP
python -m flybrain synthetic   # engine smoke test, no big download
python -m flybrain feeding     # downloads ~180 MB, reproduces Shiu et al. 2024
python -m flybrain calibrate   # find which descending neurons carry the signal
python -m flybrain play        # the game
```

The first real run downloads two files into `~/.cache/flybrain`:

* neuron annotations (Schlegel et al. 2024) from GitHub, ~32 MB
* `proofread_connections_783.feather` from Zenodo record 10676866, ~150 MB

then caches a signed sparse weight matrix as `weights.npz`.

## How it works

`flybrain/brain.py` is a leaky integrate-and-fire model with the parameters
from Shiu et al. 2024 (*Nature* 634:210): 20 ms membrane constant, threshold
at −45 mV, reset to −52 mV, 2.2 ms refractory period, 1.8 ms axonal delay,
and 0.275 mV added to the postsynaptic conductance per synapse per spike.
That last number is a jump in `g`, not in the membrane voltage: with these
time constants the resulting peak deflection in `v` is about 0.16 of it, so
one synapse is worth ~0.043 mV and roughly 160 coincident synapses are needed
to reach threshold. `python -m flybrain check` verifies this against the
closed-form solution; treating `epsp` as a direct voltage jump instead makes
every neuron ~23x too excitable and the whole brain saturates at the
refractory ceiling. A
neuron's sign comes from its predicted transmitter — acetylcholine, dopamine,
serotonin and octopamine excite; GABA and glutamate inhibit.

Each step only touches the rows of the connectivity matrix belonging to
neurons that actually spiked, which is why the whole brain runs near
real time on a laptop CPU.

`flybrain/game.py` is the interface to the world. The arena wraps at the
edges, so the fly cannot park against a wall facing outwards where it senses
nothing:

* **input** — sugar-sensing and bitter-sensing gustatory receptor neurons
  (`cell_sub_class` `sugar/water` and `bitter`, 129 and 65 cells) get
  independent Poisson drive up to 150 Hz, split left and right by where the
  food is. The drive is normalised so the strongest visible item always
  reaches 150 Hz: raw distance scaling topped out near 45 Hz, well under what
  the feeding experiment needed, and the network simply stayed silent.
  Direction is carried entirely by the left/right ratio.
* **output** — the two **DNge059** descending neurons, one per hemisphere.
  Their firing-rate asymmetry becomes a turn rate.

DNa02 is the fly's textbook steering command neuron, but it is driven by
vision rather than taste and sits silent here. `flybrain calibrate` ranks
every descending neuron by how much more it fires to sugar than to bitter,
and DNge059 comes out as the best pick: strong (274 Hz left / 271 Hz right)
and near-symmetric across hemispheres, with zero response to bitter. The
symmetry matters — a lopsided pair makes the fly circle with no food in
range. Override it with `--dn CELLTYPE`.

## Honest caveats

* The model is rate-and-wiring only. No neuromodulation, no learning, no
  dendritic compartments, no synaptic weights beyond raw synapse counts.
* Transmitter identity is a prediction, not a measurement, for most neurons.
* The sugar → ingestion motor neuron result is validated in the literature:
  this model gives 95.6 Hz to sugar, 0.0 Hz to bitter, 68.5 Hz to both.
  Steering a game with DNge059 is an extrapolation. The neuron really does
  carry a sugar signal down to the nerve cord, but whether that signal means
  "turn" in a real fly is not something this model can tell you.

## Sources

* FlyWire connectome, release 783 — https://codex.flywire.ai
* Dorkenwald et al. 2024, *Nature* — whole-brain connectome of adult *Drosophila*
* Schlegel et al. 2024, *Nature* — cell-type annotations
* Shiu et al. 2024, *Nature* 634:210 — the LIF model this follows

## Chess

`python -m flybrain.server --port 8123`, then open the page. Three modes:
**fly** (you vs the connectome), **bot** (you vs a real engine), and
**bot vs fly** (watch them play each other).

Each player has a body. The fly flies in, lands on the piece and carries it.
The humanoid walks around the board edge to the nearest point and reaches in
with a two-bone IK arm -- it has to walk, because the board is 8 units across
and a fixed stance would need a 10-unit arm to cover it. Four environments:
studio, garden, night (with a starfield) and lab.

`python -m flybrain.server` then open http://localhost:8000. A Three.js board
in the browser, the real brain on the Python side.

The fly has no board, no rules and no lookahead. Each legal move is encoded as
a stimulus -- the from-square drives one block of sensory and optic neurons,
the to-square another -- the whole brain runs for 120 ms, and the move that
drives the descending population hardest is played. The square-to-neuron
mapping is arbitrary because nothing in a fly is chess-shaped.

So the connectome genuinely picks every move, on neural excitability rather
than on whether the move is any good. The panel shows how many *distinct*
descending responses it found across the legal moves; when that is 1 the fly
cannot tell them apart and the interface says so instead of pretending. You
should win comfortably.

## Fly-brain shader for Ghostty

A custom [Ghostty](https://ghostty.org) shader that draws a glowing slice of the
real connectome behind your terminal text. Tested against Ghostty 1.3.1 on macOS (Metal).

```
ghostty/bake.py                 runs the LIF sim, picks neurons/edges, writes flybrain.glsl
ghostty/flybrain.template.glsl  the shader source (edit this one)
ghostty/flybrain.glsl           generated shader -- this is what Ghostty loads
ghostty/preview.py              offscreen renderer to PNG / GPU timing (dev only)
```

### Setup

1. `python ghostty/bake.py` (uses the cached connectome from the steps above; ~20 s).
   Re-run it any time; `-n 400` neurons, `--slow 14` display slow-down.
2. Add to `~/Library/Application Support/com.mitchellh.ghostty/config.ghostty`
   (or `~/.config/ghostty/config`), using the absolute path to your clone:
   ```
   custom-shader = /absolute/path/to/flyweb/ghostty/flybrain.glsl
   custom-shader-animation = always
   ```
3. Reload with **Cmd+Shift+,**.

Tunables are `#define`s at the top of the shader: `GLOW_STRENGTH`, `BRAIN_FIT`,
`TEXT_KEEPOUT`. Edit `flybrain.template.glsl` and re-bake (or edit the generated
file directly for a quick try).

### What you are looking at

* **Neurons and synapses are real.** 400 neurons of the FlyWire release-783
  connectome, at their real frontal (x, y) positions, levelled by principal axis.
  Lines are their strongest real synaptic connections (by synapse count). The faint
  blue haze is the density of all 139,248 neurons.
* **Spikes are real, slowed down.** `bake.py` runs the repo's LIF model on the whole
  brain under four sensory scenes (taste, sight, smell, touch), records each neuron's
  first-spike latency and spike count, and the shader replays them 14x slower. A pulse
  travels along an edge from the presynaptic neuron's spike to the postsynaptic
  neuron's spike. Firing is periodic at the neuron's mean rate, not its exact spike
  train. Between waves, faint random flicker keeps it alive.
* Colour is predicted transmitter: cyan acetylcholine, pink GABA, amber glutamate,
  green monoamines. Slightly dimmer means deeper in z.
* **Text stays readable**: glow is removed on glyph pixels and a ~2 px halo around them
  (measured against the background colour sampled from the window-padding corners; `iBackgroundColor` is not used because Ghostty leaves it at zero until the terminal state changes), and is tone-mapped to a low ceiling.
  Light-background themes are not supported (the glow is additive).

### Performance

Per-pixel work is bounded by baked grid cells (each pixel only visits the few
neurons/edges near it). Measured in an OpenGL harness on an M4: ~3.5 ms/frame at
1440x900 and ~7.5 ms at 2880x1800. Not measured inside Ghostty itself, and not on older
Airs -- if it stutters, shrink the window or lower `-n` and `--edges` in the bake.
`custom-shader-animation = always` keeps the GPU redrawing continuously.

### Verifying without Ghostty

`python ghostty/preview.py --times 1 4 8 --out /tmp/fb` renders PNGs with a fake
terminal on top (needs `pip install moderngl pillow`). `--bench` reports GPU time.

### Recording a demo GIF

1. Make the window a modest size (e.g. 1000x600) and run something with lots of text.
2. Cmd+Shift+5, "Record Selected Portion", drag over the window, record ~13 s to
   catch a full 12.8 s wave loop, stop from the menu bar.
3. Convert (`brew install ffmpeg`):
   ```
   ffmpeg -i demo.mov -vf "fps=20,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=bayer" demo.gif
   ```
