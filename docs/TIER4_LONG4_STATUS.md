# Tier4第四段续训状态

2026-09-16 15:28纽约时间（19:28 UTC）：已启动，主进程66311。
54101/54201 worker为66318/66319，54301排队；全项目947项测试通过。
本轮配方、旧实验封存哈希与源权重已复核。两支已通过初始逐局重载验证，
真实PPO更新均已达到38,912新增步；日志确认KL target0.01与KL早停生效，无失败标记。
这是启动后的即时快照，并非本轮最终结果。

- 三支各追加1,048,576步，KL target0.01及原配方保持，旧模型不覆盖。
- 从上一段固定终点权重开始，新优化器；谱系终点累计6,291,456步。
- CPU最多两个worker，整轮2小时硬上限，无自动追加。
- 方案：[TIER4_LONG4_PLAN.md](TIER4_LONG4_PLAN.md)。

新目录`runs/tier4-long4-v1`；主日志`runs/tier4-long4-v1.launch.log`，
启动记录`runs/tier4-long4-v1.launch.json`。
只有`pipeline_complete.json`及`summary.json`齐全才视为完成；
遇到`failure-*.json`停下检查。`candidate-*/updates.jsonl`记录本轮真实新增步数，
累计谱系需再加5,242,880。本轮尚无终点或测试成绩。
