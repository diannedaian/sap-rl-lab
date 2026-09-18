# Percentage rounding: per-ability repair

2026-09-07. Dodo and Badger now use floor; Crab retains ceiling. These choices
are explicit ability parameters included in the catalog fingerprint. Observed
components are regression-tested; this is not a full v0.46 parity certificate.

The frozen v7 / confirmation-v2 runtime used integer ceiling for Crab, Dodo and Badger
percentage effects. A directly inspected historical game recording contradicts
that assumption for Dodo. Do not infer that the same rule necessarily holds in
v0.46, and do not infer a shared rounding rule for every pet without evidence.

Source: [Haps' Dodo run, 4:01–4:03](https://www.youtube.com/watch?v=OahKDiOjQi8&t=241s),
listed by Grounded SAP as 29 December 2022. Browser-native playback was inspected;
no video was downloaded. Client build number is not visible.

At 241–242 seconds, the relevant pets, back to front, are:

- Level-2 Dodo 29/31, Melon.
- Level-1 Dodo 12/13, Melon.
- Level-1 Dodo 5/6, Meat Bone.
- Scorpion 1/1, Peanut.

At 243 seconds, before the first attack, they display respectively 29/31,
41/13, **25/6**, **13/1**. Crocodile separately snipes the enemy back pet.
Thus the 41-attack Dodo added **20**, not 21; the resulting 25-attack Dodo
added **12**, not 13. The attack-sharing chain uses updated attack. Meat Bone
does not enter the displayed attack transfer in this example.

This distinguishes historical behavior from ceiling, but does NOT distinguish
floor from nearest-integer/ties-to-even: 20.5→20 and 12.5→12 fit both.
Find another odd half with an odd integer part, and newer footage, before
selecting a replacement algorithm. An unsupported interpretation must not be
silently turned into a passing official-game regression.

Further observations resolve the main distinction:

- [Flame96 Dodo tooltip, 5:14](https://www.youtube.com/watch?v=F7NE_7yl41k&t=314s):
  level 3, 9 attack, explicitly 150%, **current ability value 13**. Thus
  13.5→13, not ceiling or nearest/ties-to-even. The listing says two years ago;
  exact upload and client build are not certified.
- [TheNorthernWind Badger, 6:00](https://www.youtube.com/watch?v=zziq0UPkzH0&t=360s):
  level 1, 23 attack, faint deals a visible **11** to the unprotected friendly
  neighbor (4 health→−7). Enemy mitigation is excluded from the raw-damage
  inference. The public listing says three months ago.
- [FanboyOchi Crab, 1:11](https://www.youtube.com/watch?v=vHpyTVkNphc&t=71s):
  level 1 Crab 4/1 with a highest-health friend of 6 visibly gains **+2 health**,
  becoming 4/3 before attacks. Thus Crab must not inherit Dodo's floor behavior.
  This one boundary supports the retained ceiling implementation but does not
  distinguish it from every nearest-rounding alternative. The listing says
  three months ago.

A later part of that same [Crab recording,5:48–5:50](https://www.youtube.com/watch?v=vHpyTVkNphc&t=348s)
now resolves the ceiling-versus-nearest ambiguity. At348.012s the level3 Crab
is17/14 and the four other friends have8,9,19,6health. At349.514s its ability
starts; at350.000s it visibly gains **+15** and becomes17/29, before an attack
reduces its health. Thus **19×75%=14.25→15**, not14. This matches the retained
ceiling rule. The new component regression passes without changing frozen source.
Its three key frame hashes and values are in
`crab_badger_percentage_2026.json`; screenshots are archived under
`runs/expanded-rule-video-audit-v4/`. Later damage, toys, movement, food reactions
and unsupported pets are excluded. Exact target-build certification remains open.

Selected original screenshots are archived under `runs/expanded-rule-video-audit-v2`.
Fixtures retain URLs, timestamps, values and component-only limitations. Unsupported
pets/toys and their effects are not invented in the transferred component tests.
Very low percentage results and exact target-build parity remain unobserved;
primitive boundary tests are specifications, not extra real-game evidence.

Confirmation v2 was stopped before source edits: seed1811's paired full runs
are preserved, seed1907's two partial workers received TERM, seed2027 never ran.
The old protocol/source/models remain unchanged in that directory; no test scores
were opened. A fresh fingerprint, pools and controlled training batch are required.
