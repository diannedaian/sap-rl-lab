# Eight-pet sandbox: delivery and reproduction

This is a completed small RL learning project, not a full Super Auto Pets bot.
The deliverable is the simulator, controlled experiments, a frozen checkpoint,
and a replay viewer with public source/data (the hosted instance remains private).
See [the one-page final report](ROUND5.md).

## Watch

[Open the private Replay Lab](https://sap-rl-replay-lab.dcao2028.chatgpt.site/).
Choose **Round 5 · Confirmation** for corrected Fish rules and the final experiment.
The default pair shares an initial episode seed and opponent pool: the unshaped
control times out, the swap-cost policy reaches ten wins. Different decisions
produce different later states; this selected pair is illustrative, not the
benchmark. Choose any swap-cost **Cutoff episode** to inspect remaining failures.
The new collection contains 13 cutoffs and 7 successes. For a remaining
freeze/unfreeze loop, choose `swap_cost · seed 607 / stats / #1500009`;
for a remaining swap loop, choose `swap_cost · seed 503 / stats / #1500385`.
**Round 3 · Historical** preserves the old examples and old Fish mechanics.

Use **From start**, **Play both**, **Watch endings**, or each panel's **Jump to loop**.
Click a pet for its ability. The probability panel shows what the model preferred;
it is not a probability of winning. Critic values and the dashed chart refer to
the model's *training objective*. Raw game reward is labeled separately. No
training, model inference, analytics, or cluster access runs in the website.

## Frozen artifact

The local release is `runs/round5-confirmation-v1/release/`:

Frozen candidate: **swap_cost-seed607**, selected at 750,000 continuation
decisions by validation. Fresh test: **92.20% ten-win, 1.40% cutoff**. The
all-seeds-below-1% reliability screen failed; this is an educational release,
not a production-reliable policy. It was not reselected after seeing test results.

- `model.zip`: validation-selected swap-cost checkpoint; not selected on test wins.
- `manifest.json`: exact checkpoint hash, catalog, source archive hash, protocol,
  selection rule, and inference settings.
- `../source.zip`, `../protocol.json`, `../selection.json`, `../audit.json`,
  `../summary.json`, and `../test-*.json`: provenance and full evaluation records.

These runtime artifacts are ignored by Git but published as
[GitHub Release attachments](https://github.com/diannedaian/sap-rl-lab/releases/tag/v0.1.0-eight-pet).
Follow [artifact verification and restoration](ARTIFACTS.md) before these commands
on a fresh clone. Replay data alone is not a substitute for the model or source archive.
The parent repository's default reward settings remain unchanged; explicitly
use this release for the confirmed policy.

## Run the frozen policy locally

From the repository with its existing virtual environment:

```sh
.venv/bin/python -m sap_rl_lab.training evaluate \
  runs/round5-confirmation-v1/release/model.zip \
  --opponent-league runs/round5-confirmation-v1/data/test/stats.json \
  --episodes 100 --seed 1600000 --device cpu
```

The evaluator detects the 9-input global observation and uses legal-action masks
with deterministic decisions. Reported rewards exclude swap shaping. These 100
episodes are a usage check, not a replacement for the frozen test benchmark.
The release's run manifest records the exact Python and library versions.

To reproduce the six-run study, use a **new output directory**:

```sh
.venv/bin/python -m sap_rl_lab.round5 \
  --previous-run runs/round4-two-objectives-v1 \
  --output runs/round5-reproduction-v1
.venv/bin/python scripts/audit_round5.py runs/round5-reproduction-v1
```

This requires the earlier frozen parent and pools. It runs exactly six CPU
continuations, each 1,003,520 shop decisions; no GPU is needed. It refuses to
overwrite an experiment. Same-machine deterministic replay is verified;
bit-identical retraining across different hardware/library versions is not promised.
For future source changes, reconstruct the hash-verified archived source before
reproducing historical trajectories instead of silently substituting new rules.

## Verification

The release includes Python and viewer/data test suites in CI. The independent audit checked
all 18,000 test records and exactly reproduced 90 saved-policy episodes. All 20
new viewer trajectories matched their saved test rows; the 18 old trajectories
remain unchanged. Browser checks covered collection switching, start/end,
play/pause/speed, loop jumps, keyboard stepping, full legal probabilities, battle
frames, corrected/historical Fish labels, and a visible −0.005 objective reward.
The user's current 620-pixel viewport had no horizontal overflow and no logged
browser errors or warnings. Site access was rechecked as owner-only for publishing.

The website's deployment history is separate from the public GitHub project.
Publishing source and replay data does not change the hosted instance's access settings.

## What to explain when presenting this project

1. **Environment before algorithm:** eight pets, automatic battle resolution,
   71 masked shop actions, seeded opponents, explicit terminal vs cutoff semantics.
2. **Learned component:** a small PPO actor and critic, not an LLM or foundation
   model; 199 input numbers and 38,600 parameters in the final configuration.
3. **Failure analysis:** replay inspection exposed repetitive swapping. Sampled
   training and greedy evaluation differ; zero-cost swaps and cutoff value
   bootstrapping were plausible contributors, not proven single causes.
4. **Controlled intervention:** change only swap reward, keep equal budgets and
   paired continuation seeds, select on validation, report unshaped fresh-test
   outcomes. More evaluation games do not replace independent training seeds.
5. **Honest scope:** this is evidence within one parent's continuations and three
   scripted opponent families. Fish is corrected and pet-level boundaries tested,
   but full live-game event-order parity and full-roster strength are unproven.

Stop tuning the eight-pet milestone here. The same project's ultimate goal is
still the complete Turtle Pack; follow the [expansion roadmap](ROADMAP.md) for the
next bounded phase. Larger rosters and stronger opponents should advance that
goal, not become an endless attempt to erase every eight-pet failure.
