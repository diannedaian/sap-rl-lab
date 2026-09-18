# Live human vs frozen model

Open `../play.html` through the existing site or a local HTTP server, not file://.
No Python server, GPU or cluster session is required. The first load downloads
Pyodide 0.28.3 and NumPy from the official distribution on jsDelivr. The game
then runs inside a browser Web Worker. Refreshing discards the current match.

## Provenance and verification

- Model A, seed 85101, selected by frozen validation, not by test-set results.
- `manifest.json` binds the selected checkpoint SHA, inference arrays, catalog,
  rule-source hashes and observation shape/order.
- `model.npz` contains actor and critic float32 arrays; no optimizer or pickles.
- `runtime.zip` copies nine original full-pack rule files without edits.
  Observation and post-battle settlement are AST-extracted from the frozen
  methods; `duel_runtime.py` supplies only inference and match coordination.
- NumPy/Torch parity test: 1,200 live states, all action argmaxes identical;
  maximum probability error below 1e-6, value error below 2e-6 on local CPU.
- Every browser initialization additionally checks 24 fixed observation/mask
  cases against original Torch probabilities, values and selected actions.
  Any asset digest or numerical-parity failure stops initialization.
- Original training sources, checkpoint files and historical replay data are
  unchanged. The old replay page remains at `../index.html`.

Regenerate assets from the main project with `scripts/export_duel_model.py`,
then run `tests/test_browser_duel.py` and the existing viewer tests/build.
Generated NPZ/ZIP/JSON files are reproducible inputs, not user-upload support.

## Duel contract (not a new Arena benchmark)

Two independent shops, the same 60 ordinary pets / Tier 1–6 / 18 ordinary foods.
Both retain the trained 40-round observation normalization and 30-action forced
battle rule. Each side's end-turn effects run once, then ONE shared battle is
resolved, and opposite outcomes are applied using the original settlement code.
Combat has its own seeded RNG; shopping RNGs are independent. This deliberate
head-to-head adapter is not the original asynchronous opponent-pool benchmark.

Each side begins with 5 lives and gets the original turn-3 one-life recovery.
The match ends when either side has zero lives or reaches 10 wins; 40-round
cutoffs are labeled as a limit, never a win. The AI sees its own training
observation only, not the human's shop or team. Policy inference is deterministic
masked argmax. No adaptation or training occurs from the human's games.

Battle snapshots observe existing trace boundaries without changing RNG,
events, outcome or persistent teams. Opponent actions and top-five probabilities
are revealed after the round; values/probabilities are not calibrated win odds.

The source is a frozen 0.46 executable sandbox, not verified official-client
parity. Browser-side assets (including weights) are accessible to whoever can
view the site. No secrets, absolute personal paths, private training episodes,
or optimizer state are included. Site access is managed by existing hosting.
