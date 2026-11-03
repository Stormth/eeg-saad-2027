# eeg-saad-2027

ICASSP 2027 Signal Processing Grand Challenge **GC-11: EEG-sAAD**（从脑电解码"在听谁"）参赛工程。

线性刺激重建（linear stimulus reconstruction）+ 因果 HMM 的跨数据集 AAD 基线。
完整背景、设计决策、禁忌清单见 **[AGENTS.md](AGENTS.md)**（先读它）。

## 状态

- ✅ 数据读取（extended CND）/ 线性解码器 / 因果 HMM / 评测 / 128 Hz 提交读写
- ✅ 合成数据 + 8 个自检（`python tests/test_all.py` → 8/8）
- ⏳ 5 库合并训练 → AVGC 跨库验证（P0，待真实数据）
- ⏳ 盲测提交（测试阶段 2026-11-09 开放）

## 快速开始

**在远程 SSH 服务器上复现 baseline → 看 [docs/REMOTE.md](docs/REMOTE.md)**（含数据放置、验收数字、排错表）。

```bash
# 自检（无需数据，10 秒）
python tests/test_all.py
python scripts/smoke_test.py

# 单被试快速验证（需要 data/DTU2018/dataCND/pre_dataSub{1,Stim1}.mat）
python scripts/run_linear.py --data-root data --train DTU --test DTU \
    --test-subjects 1 --out outputs/check --synthetic-switches 2

# 正式：4 库训练 + AVGC 验证
python scripts/run_linear.py --data-root data \
    --train DTU KUL UHD USTC --test AVGC --out outputs/baseline

# 盲测提交（测试阶段开放后）
python scripts/run_linear.py --data-root data \
    --test-unknown data/BLIND/dataCND --load outputs/baseline/model.npz \
    --submission outputs/submission.csv
```

依赖：`python >= 3.9`、`numpy`、`scipy`。无需 GPU。

合并多个数据集训练时默认**流式**处理（三遍扫描磁盘，内存峰值 = 单被试）；`--no-stream` 会一次性读入内存，只适合小数据。
`--prefix` 会自动探测（DTU 的 `pre_` 与 128 Hz 库的无前缀可混存）。
`--limit-per-dataset N` 用于快速冒烟。

## 结构

```
src/eegsaad/
├── cnd.py        读 .mat → Subject（含 select_channels 跨库通道对齐）
├── embed.py      时间延迟嵌入（正 lag = 未来采样，反向解码器约定）
├── linear.py     LinearDecoder：充分统计量累加 + Ledoit-Wolf 收缩
├── hmm.py        causal_hmm：2 状态、Fisher-z、只前向滤波（无平滑）
├── metrics.py    accuracy / switch_detection / latency_score / final_score
├── pipeline.py   ★ Config / train_on_subjects / run_subject / score_subject
├── registry.py   数据集别名 → dataCND 路径 / describe / 通道交集
├── submit.py     128 Hz 零阶保持写 CSV / 读回
└── synthetic.py  合成数据（共享空间模式，含 trial 内切换）

scripts/run_linear.py   主入口
scripts/smoke_test.py   10 秒冒烟
tests/test_all.py       8 个自检
```

## 数据

5 个训练集 + 2 个盲测集，下载：<https://box.nju.edu.cn/d/46b45dbcf6b5408482ca/>
（`data/` 已 gitignore，不要提交 *.mat/*.npz。）

## 铁律（详见 AGENTS.md §5/§10）

1. HMM 只做前向滤波，永不做平滑（平滑 = 违反 1 s 前瞻 = 取消资格）。
2. AVGC 只当验证集，不进训练；DTU/KUL/USTC 交叉验证数字虚高不可信。
3. 训练目标逐样本取；`expand_to_samples` 语义勿改。
4. HMM 归一化常数来自训练集；换窗长必须重算。
5. 测试集上禁止任何调参/自监督。
