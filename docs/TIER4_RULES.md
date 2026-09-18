# v6：Tier4普通商店与完整Tier5奖励依赖

目标版本0.46；2026-09-14核对。旧v5内容文件未修改，v6追加继承其定义。
50只非token宠物；普通商店最高Tier4；下一Tier奖励为Tier5，不开放普通Tier5商店。

## 来源与机制

十个宠物逐个读取Grounded SAP的**16 Mar 2026**历史行，而非搜索摘要。
这些是社区维护资料，不冒称官方客户端认证。官方Steam新闻API的可用历史
确认过罐头改为+1/+1，但本次返回没有完整0.46明细；0.46及新面包依据
[版本记录](https://superautopets.wiki.gg/wiki/Version_0.46)。

| 新奖励宠物 | 基础 | 实施规则 | 历史资料 |
| --- | --- | --- | --- |
| Armadillo | 4/8 | 开战给双方所有存活宠物8/16/24生命，含自身 | [166](https://groundedsap.co.uk/PetProfile.aspx?ID=166) |
| Cow | 4/6 | 购买替换食物（含冻结）为两份免费等级对应牛奶 | [15](https://groundedsap.co.uk/PetProfile.aspx?ID=15) |
| Crocodile | 8/4 | 末位敌人受8伤，等级增加独立触发次数，不合并成16/24伤 | [19](https://groundedsap.co.uk/PetProfile.aspx?ID=19) |
| Monkey | 1/2 | 回合结束给最前友方2/2、4/4、6/6，可作用于自身 | [46](https://groundedsap.co.uk/PetProfile.aspx?ID=46) |
| Rhino | 6/7 | 击杀后对最前敌人4/8/12伤；Tier1及token双倍，可连锁 | [55](https://groundedsap.co.uk/PetProfile.aspx?ID=55) |
| Rooster | 6/5 | 死亡召唤1/2/3只1生命Chick，继承当时攻击的一半 | [63](https://groundedsap.co.uk/PetProfile.aspx?ID=63) |
| Scorpion | 1/3 | 召唤获得Peanut；购买入队、Whale释放等走召唤事件，合并不是召唤 | [65](https://groundedsap.co.uk/PetProfile.aspx?ID=65) |
| Seal | 3/8 | 自己吃食物才给三个不同朋友1/2/3攻击 | [66](https://groundedsap.co.uk/PetProfile.aspx?ID=66) |
| Shark | 2/2 | 每个其他朋友死亡获得2/2、4/4、6/6；自身死亡不能复活 | [69](https://groundedsap.co.uk/PetProfile.aspx?ID=69) |
| Turkey | 3/4 | 朋友召唤获得3/1、6/2、9/3；商店增益永久，战斗在副本上 | [79](https://groundedsap.co.uk/PetProfile.aspx?ID=79) |

Tier4普通食物：Pear +2/+2；Bread为结束回合+7临时生命，次回合不累积；
Canned Food使当前和未来商店宠物+1/+1，包括后续奖励，不能误当队内进食。
牛奶token分别+1/+2、+2/+4、+3/+6，不参与普通刷新。
[食物资料](https://groundedsap.co.uk/Foods.aspx)；
[Cow顺序说明](https://superautopets.wiki.gg/wiki/Cow)；
[Peanut](https://superautopets.wiki.gg/wiki/Peanuts)。

Peanut只对实际造成伤害的直接攻击生效；Melon全挡住则无击杀，技能伤害不继承Peanut。
仍沿用30次普通攻击交换的规则性平局限制，其官方版本证据限制见LONG_BATTLE_RULE.md。

## 新状态与兼容

新增可序列化商店永久攻击/生命加成；队内及商店观察增加Peanut/Bread位。
动作仍139个；观察总2091维（global10、team5×130、shop9×159）。
旧v5仍1699维，不能直接载入旧网络，本轮新BC初始化。
所有新运行放新目录；旧数据和模型不覆盖，旧引擎分支与catalog保持原行为。

## 必须保留的限制

这是可执行社区规格加测试，不是完整官方客户端同局对照。继承v5已有的排序、
Whale/Parrot、百分比和30次交换上限证据限制。Rooster奇数攻击暂采用向上取整、
Crocodile重复开战事件沿用现有事件优先级、Cow奖励后替换及罐头奖励加成的
边界仍需要目标客户端fixture；测试只能证明实现符合当前明确规格。
Tier5只作为奖励并不等于完整Tier5课程；Chocolate不可得，因此Cow巧克力彩蛋
不在本轮可达状态内，不凭空开放Tier5食物。
