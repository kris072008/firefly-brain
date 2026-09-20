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
ghostty/daemon.py               live bridge: runs the sim and streams it into the shader (Phase 3)
ghostty/preview.py              offscreen renderer to PNG / GPU timing (dev only)
ghostty/selftest*.py            headless tests for the bridge and the shader decode (dev only)
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
`TEXT_KEEPOUT`, `TYPING_GAIN` (0 turns typing off), `TYPING_PULSES` (0 = cheaper). Edit `flybrain.template.glsl` and re-bake (or edit the generated
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

### Typing stimulates the brain

Every cursor move (a keypress, Enter, program output) fires the brain near the matching
spot: the terminal window is mapped onto the brain's bounding box, so top-left of your
window is the brain's top-left, and a soft bloom marks the stimulus site.

* **The cascade is the real sim.** For each cell of a 10x6 grid over the window,
  `bake.py` force-fires the real neurons nearest that spot (a short Poisson burst) in
  the full 139k-neuron LIF model and records when each of the 400 shown neurons first
  spikes (`TYP` table, 1 byte per neuron per cell). The shader replays that from the
  cell under the cursor (blended with its nearest neighbour cell), 8x slowed, with
  pulses along the real edges.
* **Stimulus strength is calibrated per cell**, not uniform: the same burst that
  triggers a brain-wide cascade in the dense centre does nothing in the optic lobes.
  The bake escalates the forced group (12 to 640 neurons, 200-500 Hz) until >= 30 of the
  shown neurons respond. Cells at the very edge may still respond weakly (min 7).
* **Ghostty limitation (stated plainly):** a shader is stateless and only receives the
  current and previous cursor position plus the time of the last change. So each keypress
  restarts the cascade at the new position, and the previous position is replayed with an
  assumed 0.15 s offset (Ghostty does not report when it was set). Slow typing or a pause
  shows the full spread; fast typing shows a rolling burst at the cursor. Cursor
  moves are the only input: keys that don't move the cursor are invisible to the shader.
* Cost: a cascade puts pulses on most edges for ~2 s, roughly +4 ms at 2880x1800 on an
  M4 in my harness (`TYPING_PULSES 0` cuts most of that).

### Live simulation bridge

Instead of replaying a baked recording, the brain on screen can be the running
simulation. A Ghostty shader cannot read files or sockets -- its only input is what is
drawn in the terminal -- so the daemon smuggles the data in through the terminal itself.

```
.venv/bin/python ghostty/daemon.py      # run this in a Ghostty window; it hosts your shell
```

Everything you type from then on happens inside the bridge (`exit` leaves it). Nothing in this
project edits your shell startup files. If you *want* it to start with every terminal, put `exec /path/to/flyweb/.venv/bin/python /path/to/flyweb/ghostty/daemon.py --rc`
at the end of your `~/.zshrc`; `--rc` makes nested shells detect the bridge and quietly start a
plain shell. Typed by hand inside a bridge, the daemon refuses and says so.
The shader config does not change. While the connectome loads (~2 s) you keep the baked
animation; when the strip appears it switches to live, and back if the daemon exits.

What drives it (all real LIF simulation, `--speed` x real time, default 0.1):
* **your typing** stimulates the mechanosensory neurons, **your terminal output** the
  visual ones, so the brain visibly responds to what you do;
* a 32 s cycle of taste / sight / touch / smell "scenes" keeps it busy when you are idle.

Things I found while tuning that you may see: the connectome model is bistable. Smell input
(or a stray spike among the strongly wired hub neurons once the network is primed) tips it
into a brain-wide self-sustaining storm that does not stop when the input does. So every
scene starts from a reset brain, smell is a short burst that is quenched mid-scene, and a
watchdog resets any storm that outlasts its input. Resting periods are genuinely quiet.

How the data gets across:
* The daemon runs your shell on a private pty **one row shorter** than the window and
  owns the last row (it also sets the scroll region so nothing scrolls into it). It repaints
  that row ~30x/s with truecolor background blocks, one column per cell. Cells 0-2 are a
  header (two magic colours and a mid-grey used to undo any gamma change in Ghostty's render
  target); the rest carry a 2-bit activity level (decayed spike trace) for each of the 400
  neurons, six per cell -- so **the window needs ~72 columns** for all of them; on a
  narrower window the missing neurons simply stay dark.
* The shader finds that row **from the cursor rectangle** (`iCurrentCursor`): its height
  and vertical position give the row grid (exact for block, hollow and bar cursors), and the
  magic colour in cell 0 confirms it. That alone is enough to **hide** the strip. Cell width
  comes from a block cursor's width, or -- for a bar cursor, which Ghostty's zsh integration
  puts at the prompt -- is measured off the strip itself by bisecting the edges of cell 0.
  So nothing depends on font size, DPI, window padding or window size, and resizes just
  work (the daemon tracks SIGWINCH and repaints the new last row; the shell is told it has
  one row fewer). It then samples one texel per neuron, undoes Ghostty's colour conversion using the
  header cells to undo Ghostty's colour conversion, hides the row, and draws the brain from the decoded levels.
* Live pulses on edges are driven by the real firing of the presynaptic neuron, but *where*
  along the edge a pulse sits is cosmetic: the strip carries firing levels at ~30 Hz, not
  sub-frame spike timing.


**Checking that live mode is really decoding.** The shader has a debug overlay (`LIVE_DEBUG`
in `flybrain.template.glsl`; edit the generated `flybrain.glsl` and reload with Cmd+Shift+,
for a quick try, or re-bake). Levels: 1 = status square, 2 = + decoded levels, 3 = + raw
diagnostics. In the top-right corner of the window:

* **green square**: strip found *and decoded* -- the brain is the live simulation.
* **orange square**: the strip row was found and hidden but the cells could not be located or
  read. You are seeing the baked animation.
* **red square**: no strip found (daemon not running, still loading, an underline cursor, or
  the window too narrow). Baked animation.
* Under the square, 64 grey squares show the decoded levels of neurons 0-63, left to right
  (black silent ... white just fired; dark blue = not decoded / not delivered).

For a decisive test run `.venv/bin/python ghostty/daemon.py --pattern` in a *fresh* window
(not inside another bridge): a wave slides along those 64 squares, ~4 s per repeat. (It sweeps
neuron *numbers*, which are not in spatial order, so the brain itself just shows scattered
flashes.)

Level 3 adds, below that, what the shader actually sees, for bug reports:
a 2x magnified copy of the strip row's raw pixels (expect magenta, green, grey, then payload
cells), and thirteen numbers, top to bottom: cursor style (0 block, 1 hollow, 2 bar, 3 underline),
cursor x, cursor y (bottom edge), cursor width, cursor height, **stage** (0 no usable cursor,
1 row not found, 2 cell width not found, 3 calibration cell unreadable, 4 live), cell width x10,
left padding x10, calibration cell brightness x1000 (~500 normally), window width, window
height, decoded columns, and colour mode (1 = texture is linear, 2 = colours were converted
to Display P3; the Ghostty default is 2).
A screenshot of that block pins down a decode failure.

Why the colours need decoding: Ghostty's Metal renderer *always* outputs Display P3. Even with
`window-colorspace = srgb` it linearises each cell colour, multiplies by an sRGB-to-P3 matrix
and re-encodes, so the texture a shader samples holds converted values (pure green reads as
about 0.46, 0.98, 0.30), and under `alpha-blending = linear` it holds linear values. The
shader recognises the header cells by hue, works out which mode it is in from the green and
grey cells, and applies the exact inverse. The tests model all four combinations
(converted or not, native or linear).

Limitations, plainly:
* **Block, hollow-block and bar cursors work.** With an underline cursor (or before Ghostty
  has drawn a cursor at all) the shader cannot locate the row: the strip stays visible and
  you get the baked animation. A hidden cursor keeps the last known geometry.
* It costs CPU: the 139k-neuron simulation takes ~35% of one core at `--speed 0.1` (about
  50% at 0.15), measured on an M4. `--speed 0.05` is much lighter and still shows cascades.
* One row of your terminal is used up, and programs that address the last row absolutely
  are told the terminal is one row shorter, so they never touch it.
* Not covered: a shell that resets the scroll region with sequences the daemon does not
  watch for; running the daemon under another multiplexer's status line.
* Verified headlessly, not in Ghostty itself: `python ghostty/selftest.py --shader` (needs
  `pip install pyte moderngl pillow`) runs the daemon in an emulated terminal (scrolling,
  Ctrl-C, alternate screen, growing/shrinking the window, cursor starting on the bottom row)
  and renders strips at five font-size / padding setups, with and without a gamma change,
  through the real shader.

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
