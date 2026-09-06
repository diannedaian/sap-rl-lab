# SAP RL Replay Lab

A static diagnostic viewer for the eight-pet SAP RL sandbox. Source and replay
data are public on GitHub; the existing hosted instance remains owner-private.
No model inference, training, analytics, or cluster connection runs in the browser.
The only hosted data is the selected replay collection and its provenance metadata.
Access protection is provided by Sites, not a hidden URL or a client-side password.

## Watch

Use **Play both**, **Watch endings**, or **Jump to loop**. Each panel can select
any episode in the selected collection. Step through shop decisions and engine-observed battle
events; click a pet for its ability. The timeline supports arrow keys.
Narrow screens stack the two panels. “All legal actions” reveals the full
masked distribution, not just the five most likely moves.

The default Round 5 collection covers all six equally budgeted confirmation
models: unshaped and swap-cost continuations, three seeds each. It includes
remaining failures and successes, plus a same-initial-seed comparison for the
validation-selected delivery model. Round 3 still contains its original 12
cutoffs and 6 ten-win successes. These are deliberately selected diagnostic
examples, not a representative sample or a fresh benchmark.
Probabilities and critic values are reconstructed using the selected checkpoints;
they are **not historical training-buffer values**. In this collection gamma is 1,
so discounting does not change the displayed realized reward sum. In Round 5,
critic and objective-return charts include the model's training swap cost;
raw game rewards are displayed separately. A single
realized return is not an expected value or the critic's training target.

Historical replays deliberately preserve the old Fish bug. The Python project's
new default catalog corrects it without rewriting Round 3 results. Round 5
training and evaluation use the corrected rules-v2.

## Develop

No third-party frontend dependencies are required. Use a modern Node.js (20+):

```sh
node --test tests/*.test.mjs
node build.mjs
python3 -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8765/ in a browser. Opening the HTML as a local file will
not load replay data reliably. Build verifies every episode before packaging.

## Provenance

Replays were exported from each experiment's hash-verified source
archive, checkpoint files, and opponent leagues. Return, number of actions,
wins, success, and cutoff status match the original saved evaluation rows.
The read-only battle observer does not consume randomness.

The Python export script and mechanics tests live in
[SAP RL Lab](https://github.com/diannedaian/sap-rl-lab).
Native emoji are used; no game art or third-party simulator code is copied.
