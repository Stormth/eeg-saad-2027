# 在远程服务器上跑 baseline

> 面向 SSH 远程机（无 GPU 需求）。全部命令从仓库根目录执行。
> 预计耗时：环境 2 分钟；单被试验证 3 分钟；5 库合并训练 1–3 小时（取决于磁盘）。

## 0. 前置检查

```bash
python3 --version        # 需要 >= 3.9
df -h .                  # 需要 >= 80 GB 空闲（zip 32 GB + 解压 ~50 GB + 产物）
nproc && free -g         # 单核单线程即可，内存峰值约 2–4 GB（流式训练）
```

## 1. 拉代码（公开仓库，免凭据）

```bash
git clone https://github.com/Stormth/eeg-saad-2027.git
cd eeg-saad-2027
```

## 2. 环境

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install numpy scipy
```

集群若有 conda / module，用 `module load python/3.10` 之类先切 Python 也行，但**一定要用 venv 隔离**，别往系统 Python 里装。

## 3. 自检（无需数据）

```bash
python tests/test_all.py     # 必须 8/8 passed
python scripts/smoke_test.py # 必须 "全部通过"
```

这两条不过就别往下走了。

## 4. 放数据

下载：<https://box.nju.edu.cn/d/46b45dbcf6b5408482ca/>（EEG-sAAD Grand Challenge Training Dataset，无需注册）

目标是这个结构（**`dataCND` 目录名不能改**，代码按它定位）：

```
data/
├── DTU2018/dataCND/pre_dataSub1.mat, pre_dataStim1.mat, ...      # 32 Hz，带 pre_ 前缀
├── KUL2016/dataCND/dataSub1.mat, dataStim1.mat, ...             # 128 Hz，无前缀
├── KUL_AAD_UltraHD2021/dataCND/...
├── USTC2026/dataCND/...
└── AVGCAAD2024/dataCND/...                                       # 只作验证集，不参与训练
```

前缀由代码自动探测（`pre_` 优先，其次无前缀），所以不同库混放不用手动指定。

验收：

```bash
PYTHONPATH=src python - <<'PY'
from eegsaad import cnd, registry
import json
print(json.dumps({k: (v["exists"], v["n_subjects"]) for k, v in registry.describe("data").items()},
                 ensure_ascii=False, indent=2))
for k in ["DTU", "KUL", "UHD", "USTC", "AVGC"]:
    ids = cnd.list_subjects("data", k)
    if not ids:
        print(k, "缺失"); continue
    s = cnd.load_subject("data", k, ids[0])
    print("%-5s %2d 被试 | %5.1f Hz | %3d 通道 | 首通道 %s" %
          (k, len(ids), s.fs, len(s.channels), s.channels[:3]))
PY
```

`registry.describe` 能列出 5 个数据集 = AGENTS.md §9 P0-1 验收通过。

## 5. 第一步：单被试验证（最重要的一步）

先确认重建的代码在真实数据上与原基线数字对得上：

```bash
python scripts/run_linear.py --data-root data --train DTU --test DTU \
    --test-subjects 1 --out outputs/check --synthetic-switches 2
```

期望（AGENTS.md §2）：

```
纯线性 SR   Acc ≈ 53.2%
+ 因果 HMM Acc ≈ 66.6%
官方同协议  53.9% → 68.4%
```

⚠️ 带 `--synthetic-switches 2` 时输出的 L/SDT/H **是合成切换**的产物，只能看 Acc，不能当成绩。

对比不带 HMM 的纯线性：

```bash
python scripts/run_linear.py --data-root data --train DTU --test DTU \
    --test-subjects 1 --out outputs/check_plain --no-hmm
```

数字对不上 → 先查 `cnd.py` 读出来的 trial 数、采样率、att 分布，再怀疑算法。

## 6. 第二步：合并训练 → AVGC 跨库验证（P0）

先用 3 个被试/库冒烟，确认通道对齐不炸：

```bash
python scripts/run_linear.py --data-root data \
    --train DTU KUL USTC --limit-per-dataset 3 --test AVGC --out outputs/smoke
```

再跑正式的（流式训练，内存峰值 = 单被试）：

```bash
python scripts/run_linear.py --data-root data \
    --train DTU KUL USTC --test AVGC --out outputs/baseline
```

- 先别把 `UHD` 放进来：它 255 通道且命名体系不同，交集会被吃掉一大截（日志里会打 `通道交集: N 通道`）。等三库跑通、看过通道名再决定。
- 产物：`outputs/baseline/model.npz`（权重 + 通道 + 标准化参数 + HMM 常数）、`report.json`（逐被试 Acc/L/H/SDT）。
- 验收：AVGC 上 Acc 与官方 53.9%（plain）/68.4%（HMM）同量级。

长任务挂后台：

```bash
mkdir -p logs
nohup python scripts/run_linear.py --data-root data \
    --train DTU KUL USTC --test AVGC --out outputs/baseline \
    > logs/baseline.log 2>&1 &
tail -f logs/baseline.log
```

（或用 `tmux new -s eeg`，断线不掉。）

## 7. 第三步：盲测提交（2026-11-09 开放后）

测试阶段最多 2 次提交、取最后一次。**提交前先本地核对格式**（§10）：

```bash
python scripts/run_linear.py --data-root data \
    --test-unknown data/BLIND/dataCND \
    --load outputs/baseline/model.npz \
    --submission outputs/submission.csv
head -3 outputs/submission.csv
# 期望表头: time_index,label,trial ；label 从 0 起；time_index 每 trial 从 0 重启
awk -F, 'NR>1{print $3}' outputs/submission.csv | sort -un | wc -l   # trial 数
```

- `data/BLIND/dataCND` 换成盲测数据的实际目录；模型文件必须是用训练集训出来的（**禁止**在测试集上再训）。
- 128 Hz 由零阶保持自动上采样；不需要手工重采样。

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FileNotFoundError: 找不到被试 N 的 dataSub/dataStim 配对` | `dataCND` 目录层级不对，或 zip 没解压完整 |
| `通道交集: 0 通道` | 两个库通道命名完全不同，先用上面第 4 节的命令打印 `s.channels` 对比 |
| `通道交集` 远小于 64 且有 warning | 正常（UHD 255 通道、 BioSemi vs NeuroScan 命名差异），缺的通道补零 |
| `KeyError: 找不到含字段 ['fs','data'] 的 struct` | `.mat` 结构与我推断的不同 → 报错里会打印实际字段名，把输出发我，改 `cnd.py` |
| Acc ≈ 50% 且 corr≈0 | 目标取了错流（att 是否 1-based？）或通道顺序错位 |
| 内存暴涨 | 用了 `--no-stream`；去掉它即可 |
| 训练很慢 | 流式训练要扫 3 遍磁盘，NFS 上更慢；可把 `data/` 放本地 SSD |

## 9. 每次改数据读取/标签对齐/因果性代码后

```bash
python tests/test_all.py                                   # 8/8
python scripts/run_linear.py --data-root data --train DTU --test DTU \
    --test-subjects 1 --out outputs/check                   # 复核 53.2 / 66.6
```

数字变了先怀疑自己，再怀疑代码（AGENTS.md §13）。
