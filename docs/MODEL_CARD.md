---
library_name: stable-baselines3
tags:
  - reinforcement-learning
  - sb3-contrib
  - maskable-ppo
  - super-auto-pets
  - game-playing
license: mit
---

# SAP RL Turtle Pack — final A/B/C policies

Three small reinforcement-learning policies for an educational, version-pinned
Super Auto Pets-inspired sandbox. All 60 ordinary Turtle Pack pets, normal Tier
1–6 shops, 18 ordinary foods and seven battle tokens are implemented. This is an
independent project, not affiliated with Team Wood Games or certified client parity.

- [Code, rules and tests](https://github.com/diannedaian/sap-rl-lab)
- [Checkpoints](https://huggingface.co/DianneDaian/sap-rl-turtle-pack)
- [Browser demo](https://sap-rl-replay-lab.dcao2028.chatgpt.site/play.html)
  (host access may require the owner's login; a local demo is included in GitHub).

## Which checkpoint?

Use **model-A-85101.zip** for the default demo. A is preferred by validation,
not cherry-picked using test results. B and C preserve the independent-seed comparison.

| Model | Validation learned pools | Held-out learned pools | New-seed confirmation |
| --- | ---: | ---: | ---: |
| A · 85101 | 83.75% | 84.7% | 80.6% |
| B · 85201 | 81.75% | 81.0% | 84.1% |
| C · 85301 | 76.75% | 81.1% | 80.9% |
| Mean | 80.75% | 82.27% | 81.87% |

These are **ten-win episode success rates**, not single-battle win rates.
Test: 2,468/3,000; confirmation: 2,456/3,000 across the three policies and two
frozen learned-opponent families. Confirmation changes episode seeds, not opponent
policies. With the three additional scripted families, both evaluations total
15,000 episodes: zero whole-episode truncations and at most 2% forced-battle
episodes in any model/family. Selection used validation only, before opening test.
The 80% criterion is a point estimate, not a confidence-interval lower bound;
shared episode seeds across models are not independent replications.

## Architecture and training

No foundation model. Separate actor/critic 64×64 Tanh MLPs, 341,516 total
parameters, 2,531 flattened observation features and 139 discrete actions.
Observations contain game/shop/team state; illegal actions are masked.

Earlier curricula used behavior cloning of heuristic demonstrations, then
Maskable PPO. Final candidates inherit Tier-5 weights and continue PPO in the
full pack. Seven independently trained opponent generators use 1,800 heuristic
demonstration episodes plus 300 validation episodes each, 20 BC epochs and
1,048,576 PPO steps each; teachers are not guaranteed optimal.

Full-pack candidate budget: 6,291,456 steps each, 18,874,368 total. Selected
checkpoints occur at cumulative 5,242,880 (A), 5,767,168 (B), 6,291,456 (C).
Opponent PPO adds 7,340,032 steps. Preparation through first passing test took
about 1h59 on local CPU, excluding earlier curricula and final confirmation.

Recipe: learning rate 3e-4, gamma 1, GAE 0.95, clip 0.2, 10 PPO epochs,
8 environments × 256 steps, batch 256, entropy coefficient 0, target KL 0.01,
and −0.005 per action. KL early stopping limits drift within PPO updates;
it is not a teacher/reference-model KL reward. Training segments carry weights
forward but create a new Adam optimizer, rather than exact optimizer resumption.

## Local inference

Install the repository and its RL dependencies; this is a custom SB3 environment,
not a Transformers model or an automatically hosted inference API.
Verified export environment: Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0,
Stable-Baselines3 2.7.1, sb3-contrib 2.7.1 and Gymnasium 1.1.1.

```bash
git clone https://github.com/diannedaian/sap-rl-lab.git
cd sap-rl-lab
python -m pip install -e '.[rl]' huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
from sb3_contrib import MaskablePPO
from sap_rl_lab.fullpack.evaluation import policy_environment_options
from sap_rl_lab.fullpack.env import SapAutoBattlerEnv

path = hf_hub_download("DianneDaian/sap-rl-turtle-pack", "model-A-85101.zip")
model = MaskablePPO.load(path, device="cpu")
env = SapAutoBattlerEnv(**policy_environment_options(model))
obs, info = env.reset(seed=42)
action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
print(env.engine.codec.describe(int(action)))
obs, reward, terminated, truncated, info = env.step(int(action))
```

The output is a discrete legal shopping decision (buy, sell, merge, freeze,
reorder, roll or end turn), not text. Action probabilities and critic values
can also be inspected; neither is a calibrated probability of winning.

## Export, safety and limitations

Public ZIPs preserve original `policy.pth`, optimizer and tensor members exactly.
Only serialized learning-rate metadata containing a personal source path is
replaced by the equivalent constant 0.0003; the derived schedule is rebuilt by
SB3 when loading. `manifest.json` records original/export hashes and validation.
Original training checkpoints stay unchanged locally. Load only trusted SB3
archives: Python deserialization can execute code.

- Frozen v0.46-inspired rules, not current official Arena or proven perfect parity.
- Reordering supports adjacent swaps, not arbitrary insertion movement. This can
  alter action costs and behavior; it is not established as the sole cause of loops.
- Shopping forces battle after 30 actions; episodes have a 40-round computational cap.
- Frozen opponent pools and a limited seed set do not prove universal generalization.
- The human-vs-bot demo is a different head-to-head format from arena evaluation.
- The author's report of being unable to beat the bot is qualitative feedback,
  not a controlled human benchmark. Reaching 80% does not prove convergence.
- No official game artwork, screenshots or client assets are included.

Project code and released weights are MIT-licensed; game names belong to their
respective owners. See the GitHub rules ledger for remaining simulator differences.
