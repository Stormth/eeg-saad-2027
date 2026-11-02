# AGENTS.md §4：时间延迟嵌入
# 约定：正 lag = 未来采样（反向解码器约定）。
# 行 i 对应判决时刻 i，用 EEG[i + lag] 构造特征（lag ∈ [0, 0.3] s → 最多看 300 ms 未来，
# 在 1000 ms 前瞻限制内，允许）。

import numpy as np


def time_delay_embed(eeg, lags, fs):
    """把 (n, C) 的 EEG 嵌入成 (n - max_lag, C*len(lags)) 设计矩阵。

    返回 (D, t0, t1, lag_idx)：
      D       设计矩阵，行 j 对应判决时刻 t0 + j
      t0      恒为 0（行对齐当前时刻）
      t1      n - max(lag_idx)
      lag_idx 各 lag 对应的采样点数
    注意：pipeline.run_subject 解包时用 `D, t0, t1, _`（AGENTS.md §6 已修坑）。
    """
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim == 1:
        eeg = eeg[:, None]
    lag_idx = [int(round(float(l) * fs)) for l in lags]
    if min(lag_idx) < 0:
        raise ValueError("负 lag 会引入平滑，违反因果（AGENTS.md 禁忌清单）")
    tmax = max(lag_idx) if lag_idx else 0
    n, c = eeg.shape
    m = n - tmax
    if m <= 0:
        raise ValueError("trial 太短，容纳不下最大 lag")
    cols = [eeg[l:l + m, :] for l in lag_idx]
    D = np.concatenate(cols, axis=1)
    return D, 0, m, lag_idx
