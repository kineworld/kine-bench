# KINEWORLD 开源 / 闭源边界（技术壁垒清单）

> 目的：明确勘境（KINEWORLD）哪些资产**公开到 GitHub（zoahdev 各仓库）**，哪些属于**公司技术壁垒、刻意不公开**。
> 原则：**架构、接口、基准、可复现评测 = 公开；具体本体的训练产物与后训练配方 = 闭源。**
> 这样既能让第三方在 KINE-Bench 上复核我们的数字、把我们的框架用作通用评测，又不泄露让 KineOne-WM 形成差异化的核心资产。

## 1. 公开范围（可推 GitHub）

| 资产 | 所在仓库 | 许可证 | 说明 |
|---|---|---|---|
| KINE-Bench 评测框架（指标全在 `kinebench/metrics.py`、`cau.py`、`emb.py`、`events.py`） | `kine-bench` | MIT | 怎么算分的全公开，欢迎挑错 |
| 适配器接口 `kinebench/adapters/*` | `kine-bench` | MIT | 任何世界模型都能接入被评测 |
| V-JEPA 2 适配器（对标 SOTA） | `kine-bench` | MIT/Apache-2.0 权重 | 见下，仅用其**公开权重**，不改其许可 |
| 合成冒烟数据 `kinebench/synth.py` | `kine-bench` | MIT | 确定性、可复现，跨模型同分布 |
| 表征健康度探针 `kinebench/rephealth.py`（有效秩 / 维度利用率 / 位移可辨识度） | `kine-bench` | MIT | **诊断手段公开，修复配方不公开**（见 §2） |
| 动作条件化 rollout 架构 `kineworld_jepa/rollout.py` | `kine-jepa` | MIT | 接口与算法结构公开 |
| 因果干预头（do(x) 接口）`kineworld_jepa/causal.py` | `kine-jepa` | MIT | 接口结构公开 |
| KineGrant v0 票据协议 `kinegrant-protocol` | `kinegrant-protocol` | MIT | 能力票据**协议**公开；注释明确"非安全控制层" |
| 站点与文档 `kineworld-site` | `kineworld-site` | MIT | 公开提案与路线图 |
| 技术雷达与采纳决策 `docs/product/KINEWORLD_TECH_RADAR_2026Q3.md` | `kine-bench` | MIT | 记录**读了什么、采纳了什么**，不含配方数值 |

## 2. 闭源范围（技术壁垒，**不推 GitHub**）

这些资产当前**不在任何 zoahdev 仓库的源码树中**（权重走 Release / 私有存储，数据走私有管线），本文件仅作为策略声明；请勿把它们加入公开提交。

| 壁垒资产 | 类别 | 为何是壁垒 |
|---|---|---|
| **KineOne-WM 最终训练权重**（非 exp001 中间检查点） | 训练产物 | 模型本身的表征质量，是产品核心 |
| **轨迹数据的动作标注**（kine-datapipe 产出的动作标签） | 数据 | 动作条件化 rollout 的监督来源，标注成本高 |
| **后训练配方**：课程调度、数据混合比例、ViT-g teacher 蒸馏配置 | 配方 | 决定最终检查点能否从 0.5 基线爬到可用 |
| **反坍缩目标的具体配方**：对比式逆动力学头的权重、warmup 调度、与正向预测损失的耦合方式、EMA 动量重调 | 配方 | 探针公开"怎么测坍缩"，配方不公开"怎么修"——这是 2026 Q3 雷达（`docs/product/KINEWORLD_TECH_RADAR_2026Q3.md` §5）的明确边界决定 |
| **具身仿真评测的私有场景集**（超出公开 MuJoCo 三场景） | 数据 | 分布外物理想象力的真实难度来源 |
| **推理 / 部署优化**（量化、蒸馏后推理图、 serving 配置） | 工程 | 产品化成本与延迟优势 |
| **客户 / 合作方的具身视频数据** | 数据 | 商业机密，且涉隐私/合约 |

> 注：`kine-jepa` 已公开的 `exp001-step5000-weights`（fp16 在线编码器，112.6 MB，MIT）是**中间检查点、刻意公开**——用于让第三方复现早期数字；最终检查点仍闭源。

## 3. V-JEPA 2 适配器的许可与边界（已核实 2026-09-01）

- **V-JEPA 2（facebook/vjepa2-vitl-fpc64-256 等）代码 MIT、权重 MIT / Apache-2.0** —— 可商用，可集成进 KINE-Bench 作为对标基线。
- **V-JEPA 1（facebookresearch/jepa）是 CC-BY-NC-4.0，禁止商用** —— 仅作为 clean-room 复现的参考，不集成、不打包其权重。
- 我们**只用其公开权重做评测**，不修改其许可、不重新分发其权重文件（由 HuggingFace / Meta 直接提供）。
- 适配器层（`vjepa2.py`）本身 MIT，可自由修改。

## 4. 提交前自检（CI / 人工）

推送前确认公开 PR **不包含**以下任一：
1. `*.pt` / `*.ckpt` / `*.safetensors`（除已声明 MIT 的公开检查点 Release 外）
2. `data/raw`、`annotations`、`labels`、`actions.json` 等私有数据目录
3. 含 `RISK_DENY`、密钥、`SECRET`、`localhost`、内网地址的配置文件
4. `requirements` 中引入 CC-BY-NC 依赖

## 5. 一句话总结

> **把"怎么评测世界模型"和"世界模型长什么样（架构）"全部开源；把"我们具体训出来的那个模型"和"怎么训出来的"留在闭源。**
