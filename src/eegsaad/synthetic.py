# AGENTS.md §4 / §5.8：合成数据。
#
# 关键设计（§6 已修坑）：所有被试共享同一空间模式 —— 每被试独立随机模式会导致
# 跨被试学不到东西（准确率永远 ~58%，重建与目标负相关 -0.63）。
#
# inject_synthetic_switches：同时交换 corr 来源（env 列）和标签，因此准确率不变、
# 只有时间结构变。任何由它得到的 L/SDT 都不能当真实成绩报告（须标 SYNTHETIC）。

import numpy as np

from .cnd import Subject


def _smooth_noise(n, rng, smooth=64):
    x = rng.standard_normal(n + 2 * smooth)
    kernel = np.hanning(smooth * 2 + 1)
    kernel = kernel / kernel.sum()
    y = np.convolve(x, kernel, mode="same")[smooth:smooth + n]
    y = y - y.mean()
    return y / (y.std() + 1e-12)


def make_dataset(n_subjects=2, n_trials=4, fs=32.0, n_samples=9600,
                 n_channels=8, noise=0.6, n_switches=1, seed=0):
    """合成跨被试数据集：共享空间模式 a1/a2，各 trial 在听流由 att 给出。"""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n_channels)
    a /= np.linalg.norm(a)

    subjects = []
    for _ in range(n_subjects):
        eeg_t, env_t, att_t, az_t = [], [], [], []
        for _ in range(n_trials):
            env1 = _smooth_noise(n_samples, rng)
            env2 = _smooth_noise(n_samples, rng)
            # 每 trial 一个真实切换（或 0），att 为逐样本 1/2
            att = np.ones(n_samples, dtype=int)
            if n_switches > 0:
                t_sw = rng.integers(n_samples // 4, 3 * n_samples // 4)
                att[t_sw:] = 2
            # EEG 由"在听包络"驱动（注意机制：单空间图 + 注意调制），
            # 与线性 SR + 相关比较的解码假设一致
            sig = np.where(att == 1, env1, env2)
            eeg = sig[:, None] * a[None, :] + noise * rng.standard_normal(
                (n_samples, n_channels))
            eeg_t.append(eeg)
            env_t.append(np.stack([env1, env2], axis=1))
            att_t.append(att)
            az_t.append(np.array([-30.0, 30.0]))
        subjects.append(Subject(fs, eeg_t, env_t, att_t, az_t,
                                ["C%02d" % (i + 1) for i in range(n_channels)],
                                source="synthetic"))
    return subjects


def inject_synthetic_switches(subject, n_switches, seed=0):
    """在 trial 内注入合成切换：交换 env 两列 + 翻转标签（从切换点起）。

    准确率不变，只有时间结构变。返回带 SYNTHETIC 标记的新 Subject。
    """
    rng = np.random.default_rng(seed)
    eeg = [e.copy() for e in subject.eeg]
    env = [v.copy() for v in subject.env]
    att = [a.copy() for a in subject.att]
    n_trials = len(eeg)
    done = 0
    while done < n_switches:
        t_idx = int(rng.integers(0, n_trials))
        n = eeg[t_idx].shape[0]
        t0 = int(rng.integers(n // 4, 3 * n // 4))
        env[t_idx][t0:, 0], env[t_idx][t0:, 1] = (
            env[t_idx][t0:, 1].copy(), env[t_idx][t0:, 0].copy())
        att[t_idx][t0:] = 3 - att[t_idx][t0:]
        done += 1
    out = Subject(subject.fs, eeg, env, att, subject.az, subject.channels,
                  source=subject.source + " [SYNTHETIC-SWITCHES]", synthetic=True)
    return out
