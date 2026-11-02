# AGENTS.md — 项目交接说明

> 给**接手这个仓库的 agent** 看的。读完这一页就能继续工作，不需要翻聊天记录。
> 人类读者也适用，但第 7、9、10 节是专门写给 agent 的操作约束。

最后更新：2026-11-02 ｜ 分支：`main` ｜ **尚未配置 git remote**（历史见 `git log`）

---

## 0. 30 秒摘要

**这是什么**：ICASSP 2027 大挑战赛 GC-11「EEG-sAAD」（从脑电解码"在听谁"）的参赛工程。
**做到哪了**：线性刺激重建 + 因果 HMM 的完整基线已实现、测试通过、在真实数据上验证过（53.2% → 66.6%，与官方 53.9% → 68.4% 吻合）。
**下一步**：把 5 个数据集合并训练，用 AVGC 做跨库验证，产出一个能提交盲测的结果。

**当前最大的风险不是算法，是时间**：开发阶段 2026-11-08 结束，测试阶段 2026-11-09 开始、12-08 截止。

---

## 1. 任务背景（为什么是这套方案）

| 项目 | 内容 |
|---|---|
| 比赛 | ICASSP 2027 Signal Processing Grand Challenge **GC-11: EEG-sAAD** |
| 平台 | Codabench <https://www.codabench.org/competitions/17856/> |
| 官网 | <https://2027.ieeeicassp.org/sp-grand-challenges/> |
| 会议 | 2027-05-16~21，加拿大多伦多 |
| 任务 | 输入多通道 EEG + 两路语音**包络**（不给原始音频），输出**每个采样点**"在听哪一路" |
| 核心难点 | **跨数据集泛化**：5 个库训练，考 2 个没见过的库（含新录的中文数据） |
| 关键约束 | 最大 **1000 ms 前瞻**；提交固定 **128 Hz 逐样本**；测试阶段最多提交 **2 次**（取最后一次）；评分容器 **600 s** 超时 |
| 评测 | `Acc`（按被试平均）、`L = exp(-SDT/60)`（切换检测）、`H = 2AL/(A+L)`（**排名只看 H**） |
| 奖励 | 前 5 名受邀投 2 页 ICASSP 论文 + 到会报告；可扩展投 OJ-SP |

### 时间线（UTC）

| 日期 | 事件 |
|---|---|
| 2026-09-08 | 开发阶段开始（已过） |
| **2026-11-08** | **开发阶段结束** |
| 2026-11-09 | 测试阶段开始，盲测集发布 |
| 2026-12-08 | 提交截止 |
| 2026-12-14 | 结果公布 |

### 数据

5 个训练集（共 31.9 GB zip），2 个盲测集：

| key | 文件夹 | 通道 | 语言 | 被试×trial | 注视受控 |
|---|---|---|---|---|---|
| `DTU` | DTU2018 | 64 | 丹麦语 | 18×60 | ❌ |
| `KUL` | KUL2016 | 64 | 荷兰语 | 16×20 | ❌ |
| `AVGC` | AVGCAAD2024 | 64 | 荷兰语 | 13×6~8 | ✅ **唯一可信验证集** |
| `UHD` | KUL_AAD_UltraHD2021 | 255（只用 64） | 荷兰语 | 30×4 | ❌ |
| `USTC` | USTC2026 | 64 | 中文 | 18×20 | ❌ |
| *盲测* | KUL-DLAAD | 64 | 荷兰语 | 19×8 | ✅ |
| *盲测* | NJU-SUSTech | 64 | 中文 | 12×— | ✅ |

下载：<https://box.nju.edu.cn/d/46b45dbcf6b5408482ca/>（无需注册即可下载）

---

## 2. 当前状态

| 项目 | 状态 | 证据 |
|---|---|---|
| 数据读取（extended CND） | ✅ 完成 | 在真实 DTU 文件上验证 |
| 线性重建解码器 | ✅ 完成 | 权重恢复测试通过，实数据 53.2% |
| 因果 HMM | ✅ 完成 | 实数据 53.2% → 66.6% |
| 评测（Acc/SDT/L/H） | ✅ 完成 | 完美预测得 L=1.0，从不切换得 L=0 |
| 128 Hz 提交读写 | ✅ 完成 | 往返测试通过 |
| 合成数据 + 自检 | ✅ 8/8 | `python tests/test_all.py` |
| CI | ✅ | `.github/workflows/ci.yml`（push 后触发） |
| **5 库合并训练** | ❌ 未做 | 只跑过 DTU 单被试 |
| **通道交集对齐** | ⚠️ 代码有（`Subject.select_channels`），**没在真数据上验证** | — |
| **盲测提交** | ❌ 未做 | 需要测试阶段数据 |
| DNN 基线对照 | ❌ 未做 | 非必需 |

### 已验证的数值（DTU 被试 1，train trial 1–50，test 51–60，32 Hz，1 s 窗）

```
纯线性 SR                 Acc = 53.2%
+ 因果 HMM                Acc = 66.6%
官方同协议参考            53.9% → 68.4%
```

加 `--synthetic-switches 2` 后：plain `Acc 53.2% / L 0.959 / H 0.684`，HMM `Acc 56.2% / L 0.875 / H 0.685`。
（切换太密会惩罚 HMM，这是合成数据的已知偏差，不可当结论。）

---

## 3. 环境

```
Python  3.9.6（系统 /usr/bin/python3 建出的 venv）
venv    ../.venv-eeg          ← 已存在，直接复用
依赖    numpy 2.0.2, scipy 1.13.1（matplotlib 装了但当前代码没用到）
GPU     不需要
```

- **不要**用系统自带的 `/Applications/DeepSeek Harness.app` 那个 Python 去装编译型包——macOS 签名校验会拒绝加载。
- `/tmp` 会被系统清理：**不要把 venv 或数据放 /tmp**（已经踩过一次，venv 的 `pyvenv.cfg` 被删导致 numpy 诡异失效）。

检查环境：

```bash
cd eeg-saad-2027
../.venv-eeg/bin/python tests/test_all.py        # 应输出 8/8 passed
../.venv-eeg/bin/python scripts/smoke_test.py    # 应输出 "全部通过"
```

### 数据现状

`data/` 里**只有 DTU 被试 1** 的两个文件（67 MB，`data/DTU2018/dataCND/pre_dataSub{1}.mat` + `pre_dataStim1.mat`）。
其余数据集**尚未下载**。`data/` 已在 `.gitignore` 中。

---

## 4. 仓库结构与职责

```
src/eegsaad/
├── cnd.py        读 .mat → Subject(fs, eeg[], env[], att[], az[], channels[])
├── embed.py      时间延迟嵌入（正 lag = 未来采样，反向解码器约定）
├── linear.py     LinearDecoder：充分统计量累加 + Ledoit-Wolf 收缩 + 求解
├── hmm.py        causal_hmm：2 状态、Fisher-z、只做前向滤波
├── metrics.py    accuracy / switch_detection / latency_score / final_score
├── pipeline.py   ★ Config 在这里；train_on_subjects / run_subject / score_subject
├── registry.py   数据集别名 → dataCND 路径
├── submit.py     128 Hz 零阶保持写 CSV / 读回
└── synthetic.py  合成数据（共享空间模式，含 trial 内切换）

scripts/
├── run_linear.py 主入口：训练 / 评测 / 盲测预测
└── smoke_test.py 10 秒跑通全流程

tests/test_all.py 8 个自检，纯 python 可跑，无需 pytest
```

约 1960 行 Python + 4 个 markdown 文档（本文件、README、docs/×2）。

---

## 5. 核心设计决策（**请勿回退**，每条都附理由）

1. **走线性路线，不上 DNN 当主线。**
   官方自己的 VLAAI 跨库只有 51.4%，低于线性基线。线性参数少、无需 GPU、无超参可调（lwcov 参数自由），在"跨库 + 有人工复现"的场景最稳。
   *反例*：如果在 DTU/KUL 上做交叉验证，CSP/空间解码会显得很好（70%+），但那是**眼动捷径**，盲测会打回 50%。

2. **AVGC 只能当验证集，不能进训练。**
   它是 5 个训练集里**唯一注视受控**的，且只有 13 人×6~8 trial，训不动但足够当"考试模拟"。
   *依据*：Geirnaert et al. 2024（arXiv:2412.01401）在同一个数据集上证明空间 AAD 类方法**达不到显著高于随机**，而线性 SR 可以跨条件/跨被试/跨数据集泛化。

3. **训练目标必须逐样本取。**
   `attendedStream` 是每个采样点一个值，一个 trial 内可能切换。
   `trial_design()` 现在用 `env[t0:t1][rows, att_seg-1]`，**不要**改回 `env[t0:t1, int(att[t0])-1]`（那是修过的 bug，会让切换后的样本全部标错）。

4. **用充分统计量，不要堆大矩阵。**
   官方日志 `6,975,446 samples × 812 columns`，天真实现需要 **45 GB**。
   `LinearDecoder.observe()` 只累加 `XᵀX`（812×812 ≈ 5 MB）、`Xᵀy`、`Σ‖x‖⁴`。**内存与数据量无关。**

5. **HMM 只做前向滤波（filtering），永不做平滑（smoothing）。**
   平滑要看未来，直接违反 1 s 规则，前 5 名被人工复现时会取消资格。
   `hmm.py` 里没有 backward 递归，这是刻意的。

6. **HMM 的归一化常数来自训练集，不能来自测试集。**
   `sigma_global` / `mean_global` 用 `normalisation_constants(训练集 corr)` 算。
   换窗长必须重算（corr 的分布随窗长变化很大）。

7. **`lookahead` 的语义：`0` = 判决只标注窗之后的样本（官方默认，严格保守）；`=window` = 判决标注自己的窗（规则允许的上限）。**
   `expand_to_samples()` 用 `b_k = clip(start_k + seg - la, 0, n)` 实现，改这个函数要重跑测试。

8. **合成切换必须标注 SYNTHETIC。**
   `inject_synthetic_switches()` 同时交换 corr 列和标签，所以**准确率不变**、只有时间结构变。
   任何用这种方式得到的 L/SDT 都不能当真实成绩报告。

---

## 6. 已修复的坑（回归测试已覆盖，别再踩）

| 坑 | 症状 | 修复位置 |
|---|---|---|
| `loadmat` 把 struct 包成 `(1,1)` object 数组 | `TypeError: float() argument must be ... NoneType` | `cnd._unwrap()` |
| `stim.names` 元素是 `['envelope_stream1']` 而非字符串 | 正则匹配不到，报"no stream entries" | `cnd._scalar_str()` |
| `stim.data` 是 `{F×N}` cell，`np.array(list).reshape` 会广播成 3 维 | `ValueError: cannot reshape 192000 into (60,2)` | 直接二维索引 `cell[i, t]` |
| `time_delay_embed` 返回 4 个值，解包顺序写错 | `TypeError: slice indices must be integers` | `pipeline.run_subject` 里 `D, t0, t1, _` |
| 合成数据每个被试随机空间模式 → 跨被试学不到东西 | 准确率永远 ~58%，重建与目标**负相关 -0.63** | `synthetic.make_dataset` 共享 `pattern` |
| 目标按 trial 取（而非逐样本） | 切换后训练信号被污染 | `pipeline.trial_design` |
| venv 放在 `/tmp` 被清理 | `ModuleNotFoundError: numpy` / 模块残缺 | 改用 `../.venv-eeg` |

**改动 `cnd.py` / `pipeline.py` 后必须跑 `tests/test_all.py`。**

---

## 7. 怎么跑

```bash
cd eeg-saad-2027
PY=../.venv-eeg/bin/python

# 1) 自检（无需数据，10 秒）
$PY tests/test_all.py
$PY scripts/smoke_test.py

# 2) 单被试快速验证（现有数据）
$PY scripts/run_linear.py --data-root data --train DTU --test DTU --test-subjects 1 \
    --out outputs/check --synthetic-switches 2

# 3) 正式：4 库训练 + AVGC 验证（需先下载数据）
$PY scripts/run_linear.py --data-root data \
    --train DTU KUL UHD USTC --test AVGC --out outputs/baseline

# 4) 盲测提交（测试阶段开放后）
$PY scripts/run_linear.py --data-root data \
    --test-unknown data/BLIND/dataCND --load outputs/baseline/model.npz \
    --submission outputs/submission.csv
```

产物：`outputs/<run>/model.npz`（权重+配置）、`report.json`（逐被试 Acc/L/H/SDT）。

常用参数：`--lags 0 0.3`、`--window 1.0`、`--hop 1.0`、`--lookahead 1.0`、`--reg lwcov`、`--no-hmm`、`--prefix pre_`（32 Hz）/ `""`（128 Hz）、`--target-fs 20`。

---

## 8. 评测口径（改代码时必须对齐）

- **Acc**：先按被试算，再对测试集内被试等权平均。
- **L**：对每个真实切换 t₀，在 `[t₀, t₀+100s]` 内找 τ\* 最大化 agreement；要求 τ\* 之后有 **≥0.5 s（64 样本 @128 Hz）连续**命中新标签；`SDT=|τ\*-t₀|`；`ℓ=exp(-SDT/60)`；**被试内取中位数**，再跨被试取均值。未检测到 = 0 分。
- **H** = `2·Acc·L/(Acc+L)`。
- 切换必须**逐 trial 统计**，不能把一个 trial 的末尾和下个 trial 的开头算成一次切换（`score_subject` 已按 trial 分组）。

---

## 9. 待办（按优先级，带验收标准）

### P0 — 让它成为一次真正的参赛（最紧急）
1. **下载并解压 5 个数据集**到 `data/`（32 GB，约 30–60 分钟）。
   验收：`registry.describe('data')` 列出 5 个数据集。
2. **通道交集对齐**跑通。`Subject.select_channels()` 已实现但**从未在真数据上验证**。
   验收：DTU(64, BioSemi) 与 USTC(64, NeuroScan) 合并训练不报错，缺失通道补零且有 warning。
3. **跑出第一个 5 库 → AVGC 的正式数字**。
   验收：`outputs/baseline/report.json` 里 AVGC 的 Acc 与官方 53.9%（plain）/68.4%（HMM）同量级；若显著低于，先查通道对齐与 z-score。

### P1 — 提分
4. `--hop 0.5` 的重叠窗实验（**注意**：重叠窗喂 HMM 会重复计证据 → 过度自信，见 `hmm.py` 注释；要么 hop=window，要么给似然做温度缩放）。
5. 多 `lags`（如 0–0.15 / 0–0.3 / 0–0.4）与多正则强度的**集成**。验收：AVGC 上 H 提升 ≥0.01。

### P2 — 可选
6. 空间解码分支（CSP / CorrCA）做融合。**必须先在 AVGC 上验证它没掉到随机**，否则不要合。
7. VLAAI DNN 对照（要 GPU，预期更差，只为写论文用）。

---

## 10. 禁忌清单

- ❌ **不要**把 `data/`、`outputs/`、`*.mat`、`*.npz` 提交进 git（已有 `.gitignore`）。
- ❌ **不要**用 DTU/KUL/USTC 的交叉验证结果下结论——没有控制注视，数字虚高。
- ❌ **不要**在测试集上做任何形式的调参/微调/自监督（规则明令禁止，前 5 名会被复现审查）。
- ❌ **不要**把 `synthetic-switches` 产生的 L/SDT 当真实成绩。
- ❌ **不要**给 HMM 加 backward 递归（= 平滑 = 违反因果）。
- ❌ **不要**用 subject/trial/dataset ID 作模型输入特征。
- ❌ **不要**在 `/tmp` 建 venv 或放数据。
- ⚠️ 提交前核对：CSV 列名 `time_index,label,trial`；`label` 从 **0** 起；`time_index` 每个 trial **从 0 重启**；采样率 **128 Hz**。

---

## 11. 外部资源

| 资源 | 地址 |
|---|---|
| 比赛主页 | <https://www.codabench.org/competitions/17856/> |
| 官网 GC 列表 | <https://2027.ieeeicassp.org/sp-grand-challenges/> |
| 数据下载 | <https://box.nju.edu.cn/d/46b45dbcf6b5408482ca/> |
| 格式手册 | 同上目录内 `extended_cnd_manual.pdf` |
| 官方线性基线（MATLAB） | 同上 `TOPUBLISH-crossDataset-example.zip`、`crossDatasetEvaluation.zip` |
| 官方 DNN 基线 | <https://github.com/SeanZhang99/SuperHugeAAD>（`python-package-working` 分支） |
| 官方 Docker | `yuanmingzhang/icassp2027eegsaadgc` |
| HMM 原始论文 | arXiv:2506.24024（Heintz, Francart, Bertrand） |
| 注视控制的关键证据 | arXiv:2412.01401（Geirnaert et al.） |

### 与本仓库的对应关系

- 官方 `defaultParams.m`：`Leeg=[0,0.3]`、`lwcov`、`targetFs=20` → 本仓库 `--lags 0 0.3 --reg lwcov --target-fs 20`
- 官方 `runHMMPostprocessor.m`：`pSwitch=0.001/s`、`meanDiff=0.2`、`sigmaGlobal=0.29`、`meanGlobal=0.02` → 本仓库 `hmm.causal_hmm` 默认值

---

## 12. 需要人类拍板的问题

1. **要不要追 DNN 路线？** 我的判断是不值得（官方 DNN 51.4% < 线性 68.4%），但若目标是发论文而不只是刷榜，可能需要一个 DNN 对照。
2. **提交策略**：测试阶段只有 2 次提交机会且取最后一次。是先交保守版（纯线性 + HMM），还是在第二次冒险上集成？
3. **是否公开代码**：Codabench 会高亮公开代码的队伍，对评奖可能有加分，但会暴露方法。
4. **远程仓库**：目前 `git remote` 为空，需要人类提供 GitHub 账号（本机无 `gh` CLI、无 token）。

---

## 13. 给下一个 agent 的建议起手式

```bash
cd /Users/tianhang/Documents/deepseek-harness/default-workspace/eeg-saad-2027
../.venv-eeg/bin/python tests/test_all.py     # 先确认 8/8
```

然后按 **第 9 节 P0** 的顺序推进。改任何涉及数据读取、标签对齐、因果性的代码后，
**必须重跑 `tests/test_all.py` 并在真实数据上复核第 2 节那三个数字**（53.2 / 66.6 / 8-of-8）。
如果数字变了，先怀疑自己，再怀疑代码。
