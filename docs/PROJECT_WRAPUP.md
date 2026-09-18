# Project wrap-up — 2026-09-18

**The educational 60-pet project is complete.** No further training is being
started for this release. The earlier eight-pet release and all local selected
models, experiment directories and original checkpoints remain intact.

The final trio achieved **82.27%** mean ten-win success against the frozen held-out
learned pools; one fresh-episode-seed confirmation yielded **81.87%**. This is not
a claim about real Arena or human win rates. The user has played the browser bot
and reports being unable to beat it—a useful demo outcome, separate from evaluation.

## Deliverables

- [GitHub source, tests and experiment history](https://github.com/diannedaian/sap-rl-lab)
- [Public final A/B/C checkpoints](https://huggingface.co/DianneDaian/sap-rl-turtle-pack)
- [Model card and inference instructions](MODEL_CARD.md)
- [Playable browser model](https://sap-rl-replay-lab.dcao2028.chatgpt.site/play.html)
  and a local browser build in `viewer/`
- [Compact final evidence with original/public checksums](evidence/fullpack-v1/manifest.json)
- [Validation curves](figures/fullpack-final/validation.png) and
  [opponent BC curves](figures/fullpack-final/bc.png)

The historical [delivery report](FULLPACK_DELIVERY.md) is preserved byte-for-byte
because it participates in the original audit. Its local `runs/` links and
then-current "not uploaded" wording describe that earlier snapshot. The compact
public evidence and Hugging Face release above are the new distribution layer;
they do not replace or retroactively rewrite the experiment.

## What this project demonstrates

1. Building versioned game rules, action masks, regression fixtures and replays
   before interpreting training results.
2. Diagnosing degenerate behavior rather than treating reward curves as game strength.
3. Bootstrapping with heuristic behavior cloning, then improving with PPO.
4. Controlling update size with target-KL early stopping, and using validation
   checkpoint selection rather than assuming the last checkpoint is best.
5. Separating training/validation/test opponent pools, reporting multiple seeds,
   and checking a fixed model selection on fresh episode seeds.
6. Exporting the policy to a small browser demo with Torch/NumPy inference parity.

## Scope boundary

All 60 pets does not mean complete official-client parity. Adjacent-only swapping,
the 30-action shopping budget and 40-round episode cap remain explicit differences.
Insertion movement would change the environment/action contract and merit a new
version plus retraining; it is deliberately not slipped into a finished release.
No causal claim is made that movement alone caused the earlier loops.

Future work, if desired, is broader opponent distributions, controlled human
evaluation, and targeted client-parity fixtures—not another unbounded training run.
The current evidence supports delivery, not a proof of optimality or convergence.

## Release verification

Local wrap-up verification passed 1,039 Python tests, 46 viewer tests, the browser
production build, lint and formatting checks (three hash-frozen historical helper
files are intentionally excluded from formatter changes). All three public model
exports passed archive privacy checks, byte-identical tensor/optimizer checks,
and 128 legal inference steps matching the original checkpoints. The public
evidence retains numeric rows and records original versus redacted checksums.

Publication status: the public Hugging Face repository has been created; its
file upload is pending Chrome's user-controlled file-access permission. The
prepared exports are not yet downloadable. No additional training or deletion
is part of this wrap-up.
