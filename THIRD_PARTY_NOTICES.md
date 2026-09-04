# Third-party design references

This project is a clean implementation. It does not vendor code or game assets
from the projects below, but it deliberately preserves several of their solid
architectural ideas.

- [`manny405/sapai`](https://github.com/manny405/sapai), MIT: separate game
  objects, explicit trigger phases, serializable state, deterministic battle
  traces, and tests for ability interactions.
- [`andreped/sapai-gym`](https://github.com/andreped/sapai-gym), MIT: a standard
  RL environment, stable action identifiers, replaceable opponent generators,
  and invalid-action masks.
- [`andreped/super-ml-pets`](https://github.com/andreped/super-ml-pets), MIT:
  Maskable PPO training, scripted baselines, checkpoints, and evaluation hooks.

Game names and mechanics are used only as factual compatibility references.
No Super Auto Pets art, audio, or proprietary game code is included. The
initial rules were checked against the community-maintained Super Auto Pets
Wiki in September 2026. They must be validated against the target game version
before claiming exact game parity.

