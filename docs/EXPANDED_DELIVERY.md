# 30 宠 expanded 沙盒：交付结论

2026-09-07。**已通过预先约定的训练稳定性验收。** 范围没有缩小：Tier 1–3
共30只宠物都有技能；普通商店封顶Tier2，Tier3用于升级奖励和Spider。
这不是完整60宠Turtle，也不声称与官方客户端逐帧完全一致。

## 结果

六组从零训练，每组8,388,608次决策；三个独立种子，各有配对无扣分对照。
处理组为每步扣0.005，不是只扣换位。验证集选模型、冻结后才打开测试集。

| 种子 | 脚本对手十胜：对照 → 扣分 | 学习型对手十胜：对照 → 扣分 | 扣分组脚本强制介入 |
| --- | --- | --- | --- |
| 1811 | 92.47% → 95.57% | 36.8% → 54.1% | 0 / 3,000 |
| 1907 | 94.37% → 94.90% | 44.0% → 40.0% | 0 / 3,000 |
| 2027 | 89.30% → 94.83% | 32.8% → 39.7% | 2 / 3,000 |

每个模型测试三个脚本家族各1,000局，另测学习型对手1,000局。三个扣分模型
均无截断；每个脚本家族强制介入率最高0.1%，低于1%门槛；十胜率均超过90%，
且没有超过对照1个百分点的退步。整数计数复核与原验收一致。
三个脚本基线的脚本十胜率为51.53% / 25.00% / 46.00%，学习型挑战为
6.7% / 0.8% / 5.5%。这是冻结沙盒对手分布上的结果，不是人类竞技排名。

**交付模型固定为 `action_cost-seed1907`**：由验证集选出，不因1811测试更高
而换模型。它在4,000局正式测试中没有强制介入或截断；学习型挑战40%，说明
泛化还有明显空间。每步扣分在这三个种子的脚本测试中都提高十胜率并减少
动作，但不能宣称它对所有对手、超参数或完整游戏都更好。

## 验证与剩余限制

- 完整615项测试通过；6,000场混合战斗重复复现、400局完整回放一致，
  200个可达状态的3,458个合法分支各执行两次，旧八宠300条记录仍一致。
- 已核对真实录像中的升级奖励、开始回合、死亡/召唤、致死后触发及百分比取整。
  修复了包括Dodo/Badger向下取整、Crab向上取整等问题；具体证据及版本限制见
  [真实案例](REAL_GAME_CASES.md)和[规则表](TIER12_RULES.md)。
- 六个模型均精确重现600条验证记录；交付模型还用归档源码独立重现了600条。
- 最终检查20局失败、12局成功，以及所有3局候选模型强制介入案例。
  强制进战斗后的终止标志正确。**仍有多余换位/冻结和价值估计误差**，不是
  完美策略；没有把零截断等同于零循环。详见[回放审查](REPLAY_INSPECTION.md)。
- 录像不是精确0.46客户端认证；部分拥挤库存、重复攻击目标死亡后的调度等边界
  仍仅有规格测试。30动作强制战斗是工程设计，30次战斗交换上限有历史录像
  支持但仍标注为版本待确认。当前没有已知未修复的重大规则矛盾，不保证绝对无bug。

现在适合冻结这个学习里程碑，不再继续刷同一测试集。下一阶段若开放普通Tier3
商店，需要一起补齐真实Tier4升级依赖，并扩展对手池；最终目标仍是完整Turtle。

## 本地复现

所有结果保留在 `runs/expanded-confirmation-v3/`，模型在
`action_cost-seed1907/best_model.zip`，同目录有运行配置。没有上传新版本、删除
旧实验或改动已发布八宠默认规则。

在仓库根目录运行；下面的`recheck`目录/文件必须尚不存在，只加载本项目可信模型：

```sh
# 完整测试
.venv/bin/python -m pytest -o addopts='' -q

# 用冻结的归档源码重放交付模型验证集；不会训练或改写原结果
MPLCONFIGDIR=/private/tmp/sap-rl-mpl-cache .venv/bin/python \
  scripts/reload_archived_pilot.py \
  --run runs/expanded-confirmation-v3 --arm action_cost-seed1907 \
  --output runs/expanded-delivery-recheck

# 重建同一组最终诊断回放；不会重新选模型
MPLCONFIGDIR=/private/tmp/sap-rl-mpl-cache .venv/bin/python \
  scripts/inspect_confirmation_delivery.py \
  --confirmation runs/expanded-confirmation-v3 \
  --output runs/expanded-replays-recheck
```

本机已实际执行前两类检查，归档重放证据在
`runs/expanded-confirmation-v3-archived-delivery-reload-v1/summary.json`；第三类已在
`runs/expanded-confirmation-v3-delivery-replays-v1/`执行并有单独人工审查记录。
运行环境及依赖版本在模型的`run_manifest.json`中。

模型SHA256：`3c1f3ccc16db284c789d4d7a6f92d39561c332fca4693d1879795785e6c94d39`。
源码ZIP：`78db9add1ca353959e2f1b63f389385fb996ed50a43cf1118a150c553520b38d`。
协议：`e318a5cbb7b1c0c7fc58071f907486db02fb8fba401cc35df24cb2f9db83d108`。
