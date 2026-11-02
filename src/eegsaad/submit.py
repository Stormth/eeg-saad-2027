# AGENTS.md §4：128 Hz 零阶保持提交读写。
#
# 提交格式核对（AGENTS.md §10）：
# - CSV 列名 time_index,label,trial
# - label 从 0 起
# - time_index 每个 trial 从 0 重启
# - 采样率 128 Hz（其他采样率的逐样本判决用零阶保持上采样）

import csv

import numpy as np

SUBMISSION_FS = 128.0


def zoh_upsample(labels, source_fs, target_fs=SUBMISSION_FS):
    """零阶保持：把 source_fs 的逐样本标签重采样到 target_fs。"""
    labels = np.asarray(labels)
    if int(source_fs) == int(target_fs):
        return labels.astype(int)
    n_out = int(round(len(labels) * target_fs / source_fs))
    idx = np.minimum((np.arange(n_out) * source_fs / target_fs).astype(int),
                     len(labels) - 1)
    return labels[idx].astype(int)


def write_submission(path, trial_labels, source_fs):
    """trial_labels: 每 trial 一个 0/1 逐样本标签数组（source_fs 采样率）。"""
    rows = []
    for trial_idx, lab in enumerate(trial_labels):
        lab = zoh_upsample(lab, source_fs)
        for t, v in enumerate(lab):
            rows.append((t, int(v), trial_idx))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_index", "label", "trial"])
        w.writerows(rows)
    return len(rows)


def read_submission(path):
    """读回提交文件，返回 [np.ndarray(labels), ...]（按 trial 分组）。"""
    trials = {}
    header = None
    with open(path) as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if header is None:
                header = [c.strip() for c in row]
                if header != ["time_index", "label", "trial"]:
                    raise ValueError("提交 CSV 列名不对: %s" % header)
                continue
            t, label, trial = int(row[0]), int(row[1]), int(row[2])
            trials.setdefault(trial, []).append((t, label))
    out = []
    for k in sorted(trials):
        items = sorted(trials[k])
        out.append(np.array([v for _, v in items], dtype=int))
    return out
