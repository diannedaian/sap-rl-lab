# Play against the delivered 60-pet model

Published 2026-09-18:
https://sap-rl-replay-lab.dcao2028.chatgpt.site/play.html

The existing replay viewer remains at `/index.html`, with a link to live play.
Sites version 3, source commit `ef80c9644edf6718fc591d268d90bead4c9e0c4c`.
Deployment reported succeeded. Existing owner-only access was preserved; the
live browser reaches the ChatGPT sign-in gate. Hosted post-login play has NOT
been verified. A proposed owner-token asset check was rejected by the access
reviewer and was not executed; no alternate bypass was attempted.

## Delivered behavior

Human-operated shop versus deterministic inference from model A (85101), the
frozen validation-selected full Tier 1–6 checkpoint. Every round uses one shared
combat result, opposite outcomes, original end/start-turn effects and lives.
Buy, feed, shop/team merge, sell, adjacent swap, freeze, reroll, end turn;
30 shopping actions force combat. Up to 40 rounds, zero-life loss terminates.
This direct duel is a different opponent format, not another 80% Arena test.

All 60 ordinary pets and 18 ordinary foods use original rule files. Browser
execution uses Pyodide 0.28.3, NumPy and a Web Worker; no GPU, server Python,
training, optimizer loading or cluster connection. The page reveals the model's
last-round actions, top-five probabilities and critic values after battle.
Current model shop is hidden; no human board is added to its observation.

Original training code and checkpoints were not modified. Only new export/test
files and the existing viewer were changed. No main-repository commit or GitHub
push was performed by this feature task; deployment uses the site's own repo.

## Verification

- 12 Python tests passed (9 new duel cases plus 3 existing viewer checks).
- 46 existing frontend/replay tests passed; viewer build verifies 38 replays.
- NumPy inference matches Torch action choices in 1,200/1,200 live states;
  max probability error 8.94e-7, max value error 1.19e-6.
- Every browser initialization checks 24 golden Torch cases and asset hashes.
- Local browser: initialization, purchase, shop merge, Honey target selection,
  actual model battle, start/step replay, probability inspection and next-round
  gold/team persistence verified. Desktop layout visually inspected.
- Integration checks: stale/illegal actions, frozen shop item persistence,
  forced combat, full-match terminal state, three deterministic self-play
  matches, combat observer RNG/result parity, original settlement parity.
- Final source lint and JavaScript syntax checks passed. Frozen rule files are
  byte-for-byte compared by tests; no official-client parity claim is made.

Implementation/provenance: `viewer/duel/README.md`.
Export: `scripts/export_duel_model.py`; regression: `tests/test_browser_duel.py`.

The site remains private and requires its owner's ChatGPT sign-in. Opening
the public URL does not by itself authorize bypassing that sign-in gate.
