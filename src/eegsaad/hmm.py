# AGENTS.md §4 / §5.5 / §5.6：因果 HMM 后处理
# 参考论文 arXiv:2506.24024 (Heintz, Francart, Bertrand)。
#
# 铁律：
# - 只做前向滤波 (filtering)，永不做平滑 (smoothing)。平滑要看未来，
#   违反 1000 ms 前瞻规则，前 5 名被人工复现时会取消资格。
#   本文件刻意没有 backward 递归 —— 不要加回去。
# - 归一化常数 (sigma_global / mean_global) 必须来自训练集 corr，
#   不能来自测试集。换窗长必须重算。

import numpy as np

# 官方 runHMMPostprocessor.m 默认值
DEFAULT_P_SWITCH = 0.001     # /s
DEFAULT_MEAN_DIFF = 0.2
DEFAULT_SIGMA_GLOBAL = 0.29
DEFAULT_MEAN_GLOBAL = 0.02


def fisher_z(r, eps=1e-6):
    r = np.clip(np.asarray(r, dtype=np.float64), -1.0 + eps, 1.0 - eps)
    return np.arctanh(r)


def normalisation_constants(corr, attended):
    """由训练集相关系数算 Fisher-z 差值的 (mean, sigma)。

    corr     : (n_windows, 2)
    attended : (n_windows,) 0/1，训练时的真实在听流
    返回 (mean_global, sigma_global)：在听流与另一流的 Fisher-z 相关之差的分布参数。
    """
    z = fisher_z(corr)
    attended = np.asarray(attended).astype(int)
    d = z[np.arange(len(attended)), attended] - z[np.arange(len(attended)), 1 - attended]
    mean = float(np.mean(d))
    sigma = float(np.std(d))
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = DEFAULT_SIGMA_GLOBAL
        mean = mean if np.isfinite(mean) else DEFAULT_MEAN_GLOBAL
    return mean, sigma


def _norm_logpdf(x, mean, sigma):
    return -0.5 * ((x - mean) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))


def causal_hmm(corr, mean_global, sigma_global, dt, p_switch=DEFAULT_P_SWITCH,
               temp=1.0):
    """2 状态因果 HMM 前向滤波。

    corr   : (n_windows, 2) 每个判决窗内重建信号与两路包络的相关
    dt     : 相邻判决窗的间隔（秒），通常 = hop
    temp   : 似然温度。hop < window 时窗会重叠、证据被重复计数 → 过度自信
             （AGENTS.md §9 P1-4），此时应给 temp > 1 做缩放。
    返回 (decisions, posteriors)：decisions (n_windows,) 0/1，posteriors (n_windows, 2)。
    """
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[1] != 2:
        raise ValueError("corr 形状须为 (n, 2)")
    z = fisher_z(corr)
    d0 = z[:, 0] - z[:, 1]   # 假设状态 0（听流 1）时的差值
    d1 = z[:, 1] - z[:, 0]   # 假设状态 1（听流 2）

    loglik = np.stack(
        [_norm_logpdf(d0, mean_global, sigma_global),
         _norm_logpdf(d1, mean_global, sigma_global)],
        axis=1,
    ) / float(temp)

    p01 = 1.0 - np.exp(-float(p_switch) * float(dt))  # 每步切换概率
    p01 = min(max(p01, 1e-12), 0.5)
    trans = np.array([[1 - p01, p01], [p01, 1 - p01]])

    n = corr.shape[0]
    posterior = np.array([0.5, 0.5])
    decisions = np.zeros(n, dtype=int)
    posts = np.zeros((n, 2))
    for t in range(n):
        posterior = posterior @ trans  # 前向预测（只用过去 + 当前，无未来）
        joint = posterior * np.exp(loglik[t] - loglik[t].max())
        s = joint.sum()
        if s <= 0:
            joint = posterior
            s = joint.sum()
        posterior = joint / s
        posts[t] = posterior
        decisions[t] = int(np.argmax(posterior))
    return decisions, posts
