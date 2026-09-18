# Long-battle draw: evidence and provisional implementation

2026-09-07. **The precise v0.46 client cutoff is not verified.** This remains a
rule-provenance limitation, not something a high training score can resolve.
There is now a frame-reviewed historical game recording showing thirty exchanges
and post-attack summons before the draw; see below. No training rule changed.

## Failure found

Five 1-attack / 50-health Swans versus the same team raised a RuntimeError after
200 attack exchanges. These are supported pets with legal stats. Uniform random
stress previously reached only 35 exchanges and missed this boundary. Confirmation
v1 was stopped with all partial artifacts preserved and no held-out scores opened.

## Evidence

- [April 2024 firsthand report](https://www.reddit.com/r/superautopets/comments/1c542fq/):
  an unfinished long battle automatically drew. Comments suggest approximately 30.
- [March 2023 firsthand recording post](https://www.reddit.com/r/superautopets/comments/11otjry/):
  reports a long battle ending in a draw with pets remaining; no exact count
  established from the retrieved text. Its video has not been frame-counted here.
- [March 2022 discussion](https://www.reddit.com/r/superautopets/comments/tqkeme/)
  and [April 2023 discussion](https://www.reddit.com/r/superautopets/comments/12f9hs9/)
  describe a 30-attack cutoff. These are community claims, not official constants.
- [Conflicting November 2022 claim](https://www.reddit.com/r/superautopets/comments/yj3i72/)
  says more than 40 attacks. This conflict remains unresolved.
- Public official Steam announcements inspected on this date do not identify
  the threshold. Neither the inspected sapai battle loop nor SuperAutoTest's
  combat source supplied a verified original-game constant.

### Observed historical recording: thirty exchanges, then draw

[Skoottie's actual gameplay](https://www.youtube.com/watch?v=Z2xB3Sgo6Uk&t=563s),
published 2022-03-29, explicitly describes the new-pack **test server**. At
9:26–10:07 the front Tapirs repeatedly attack, faint and summon replacements.
The reviewed count is **30 simultaneous front-pet exchanges**. At 10:08 both
replacement front pets are alive and the opponent's summon buff has resolved;
at 10:13 the Draw overlay appears, with eight trophies and six lives unchanged.
At 10:20 the next shop is turn 13. This is stronger than a comment guessing a
number, but it is historical test-server evidence, not an observed v0.46 build.

Method: bounded browser playback screenshots, approximately 0.2 seconds apart,
with visual review of the attack candidates. A rough red-pixel detector returned
31 candidates: two were orange game-menu text and a real exchange was merged
into the second menu pulse. Both menu images were rejected visually; the missing
exchange at approximately 578.44 seconds was separately replayed and inspected.
All thirty attack images were checked. Thresholds 150/250/400/500 also extract
the same thirty pulses from the recorded screenshot metrics. These thresholds
locate video frames; they are not game rules or RL hyperparameters.

Playback failed late in the first pass, so 606–613 seconds was separately played
through the visible Draw result. The short overlap contains the same final
exchange and no extra attack. This avoids inferring the result from a failed
player. Screenshots/metrics are retained in `runs/expanded-rule-video-audit-v1`;
the source, thirty approximate timestamps, result and exclusions are recorded in
`tests/fixtures/real_game/long_battle_tapir_2022.json`.

The new regression transfers only the generic count and summon-before-draw
component onto supported Cricket/Swan teams. It does **not** replay unsupported
Tapir/other pets or certify the entire recorded battle. Natural win/loss priority
on exchange 30 and the exact v0.46 constant remain unobserved boundary details.

## Working interpretation, not certification

The development catalog provisionally uses **30 simultaneous front-pet attack
exchanges**, based on the repeated community description. It is not an RL
hyperparameter chosen for a higher score, and it is not the 30 shop-action cap.
After the final allowed exchange, resolve all triggered abilities and summons.
If a natural win/loss/draw has occurred, retain it; otherwise draw with survivors.
Do not award a win, lose a life, or terminate the entire episode for this draw.

The threshold is a validated, catalog-fingerprinted field. Old unconfigured
catalog fingerprints and the released eight-pet resolver are unchanged. The
technical event-queue exception is still an exception, never disguised as a draw.

The legal long-battle crash is repaired under this explicit interpretation, but
exact target-version parity remains open. A countable historical recording now
supports thirty and post-attack summons, but a target-version boundary check is
still needed before exact parity sign-off. If it contradicts this interpretation,
change the versioned rule and
rerun affected experiments; do not relabel the old scores.

Boundary unit tests remain specification tests; the separate historical fixture
supports only its explicitly observed count/ordering component.
Stress should include low-attack/high-health and summon-heavy teams, not only
uniform random stats. Track these draw events separately from forced shop endings.

## Predeclared sensitivity check

After a v7 pilot arm completes, `scripts/audit_battle_limit_sensitivity.py` can
reload its frozen selected model on the same validation seeds/pools at limits
30, 40 and 50. The 30 case must first exactly reproduce its saved validation
rows. Report every changed episode row and all three sets of scores, without
selecting a rule, a checkpoint or new hyperparameters from them. This opens no
held-out test and performs no training. It measures the conditional effect on
this validation sample, **not** which cutoff the official game uses.

Executed for all three completed v7 models. Each model first
exactly reproduced all 180 selected validation episode rows at 30; changing the
limit to 40 or 50 changed no episode rows. None of the selected validation suites had
an attack-limit draw. Results are in `runs/expanded-pilot-v7/sensitivity-control/`
and `sensitivity-action-cost/`, plus `sensitivity-swap-cost/`, including model,
protocol and audit-script hashes.
This does not resolve the original-client threshold or certify identical event
traces. No rule or checkpoint was selected from these counterfactual scores.
