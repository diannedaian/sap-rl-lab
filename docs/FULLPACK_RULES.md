# 完整包规则实施账本（0.46，开发中）

2026-09-17。新代码在`src/sap_rl_lab/fullpack`；旧命名空间与所有旧catalog不改。
v7为Tier1–5普通商店，v8为Tier1–6普通商店；两者都有60只非token宠物，
18种普通可刷新食物及相关召唤/食物token。继承v6已经声明的客户端实证限制。
代码通过规格测试不等于官方同局对照；正式训练门还包括压力、可达回放和PPO重载。

## 新宠物资料

以下十项均于本日实际读取Grounded SAP的 **16 Mar 2026** 历史行。
它是社区维护的开发者数据展示，不是官方客户端执行轨迹；不能混入当前新版本改动。

| 宠物 | 基础 | 本版规格 | 历史页面 |
| --- | --- | --- | --- |
| Boar | 10/6 | 攻击前自身+4/2、8/4、12/6；影响本次攻击 | [103](https://groundedsap.co.uk/PetProfile.aspx?ID=103) |
| Cat | 4/5 | 商店数值食物倍率2/3/4，每回合两次；多Cat额外倍率相加 | [11](https://groundedsap.co.uk/PetProfile.aspx?ID=11) |
| Dragon | 3/8 | Tier1朋友购买后，其他朋友+1/1、2/2、3/3；每回合四次 | [25](https://groundedsap.co.uk/PetProfile.aspx?ID=25) |
| Fly | 4/4 | 朋友死亡位置召唤4/4、8/8、12/12 Zombie Fly；每回合三次 | [30](https://groundedsap.co.uk/PetProfile.aspx?ID=30) |
| Gorilla | 7/10 | 受伤后获得Coconut；每回合1/2/3次 | [36](https://groundedsap.co.uk/PetProfile.aspx?ID=36) |
| Leopard | 10/4 | 开战对1/2/3个不同随机敌人造成自身攻击一半伤害 | [41](https://groundedsap.co.uk/PetProfile.aspx?ID=41) |
| Mammoth | 4/12 | 死亡后其他朋友+2/2、4/4、6/6 | [45](https://groundedsap.co.uk/PetProfile.aspx?ID=45) |
| Piranha | 10/4 | 受伤时其他朋友+3/6/9攻击，致死伤害仍算受伤 | [118](https://groundedsap.co.uk/PetProfile.aspx?ID=118) |
| Snake | 8/3 | 前方朋友攻击后随机敌人受5/10/15伤，每场五次 | [73](https://groundedsap.co.uk/PetProfile.aspx?ID=73) |
| Tiger | 6/4 | 仅战斗中，前方朋友能力以Tiger等级重复；不重复perk | [77](https://groundedsap.co.uk/PetProfile.aspx?ID=77) |

Tiger页面列有2025-12-19限次修复；实现不能通过repeat绕过原拥有者的限次。
Wiki同页存在相互矛盾的新旧说明：旧段落称重复不扣次数，但明确标注2026-01-08的
新说明改为消耗次数；本项目锁定0.46，采用后者并测试耗尽时不再重复，
见[Tiger版本说明](https://superautopets.wiki.gg/wiki/Tiger)。不能照抄未注明年代的旧段落。
Tiger不重复另一只Tiger/复制Tiger的Parrot。Tiger+Whale吞两个前方朋友的说明也已读取。
Fly不因Zombie Fly死亡而触发；满队不浪费次数；商店消耗的Fly/Gorilla次数带进本场战斗，
Snake/Hippo的每场次数在开战重置。相关说明见[Fly](https://superautopets.wiki.gg/wiki/Fly)。

## 食物与状态

Tier5：Chili、Chocolate、Sushi；Tier6：Melon、Mushroom、Pizza、Steak。
Chocolate每点经验同时增加+1/+1（满等级也有数值收益），不复制另一只宠物的属性或perk；
升级仍给正确下一阶（封顶Tier6）奖励，见[经验机制](https://superautopets.wiki.gg/wiki/Experience)。
Cat按一份食物而不是每个随机收件人扣次数；不放大XP或perk。
Cow吃Chocolate产生对应等级Chocolate Milk，发生在升级奖励摆放之前。
依据：[食物表](https://groundedsap.co.uk/Foods.aspx)、[Cow顺序说明](https://superautopets.wiki.gg/wiki/Cow)。

Coconut完全挡下一次伤害，不触发Hurt，也阻止Peanut击杀；Steak仅第一次攻击额外20伤；
Mushroom重召同等级的1/1新实例，不保留Mushroom本身或旧计数。
新增perk均可观察；次数观察不再把4次/5次都压成相同值。

## 已执行的审计与尚存限制

- 60项新版规则/编排定向测试通过，其中包括十只新增宠物的三级能力、
  Tiger+Whale/Crocodile/Snake/Fly、Chocolate满级与Cow升级、Cat叠加及额度。
- 300组只含旧物种、无新增机制的战斗与旧引擎对照，胜负及战斗结算一致；
  排除明确修正的Rat敌方召唤通知，以及可能复制Rat的Parrot。日志文本不作跨版本一致要求。
- 开发压力检查完成6,000组战斗各复现两次、160局按动作精确重放；
  检查状态、观察、动作mask和奖励一致。这是自一致性检查，不是官方客户端对照。
- 32步真实PPO更新后保存/重载，4局验证动作与结果逐行一致；已验证学习型快照生成。
  正式训练前还会针对最终封存源码分别重跑Tier5/6训练与压力门，并跑完整回归套件。

- Tiger与同时致死、复制能力的极端精确优先级仍缺客户端录制证据；
  Leopard奇数攻击向上取整暂作为明确规格，仍需目标客户端fixture验证。
- Mushroom、Fly、原自身召唤、Whale释放的组合测试覆盖常见情况，不能证明所有连锁都正确。
- 食物、升级奖励、触发次数刷新以及随机Buff优先非满属性仍可能存在未覆盖的极端交互。
- 继承旧商店/结算/30攻击交换上限的证据限制；不能声称官方完整同局一致。
- 随机和老师完整局已见到全部60只非token宠物；随机生成战斗见到全部67种含token物种。
  出现过不等于每种技能在每种上下文都覆盖，不能将物种覆盖率包装成语义正确率。

当前仅机制单元测试通过不构成训练门通过；全部训练必须记录准确版本和代码hash。
