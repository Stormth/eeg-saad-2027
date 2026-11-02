# AGENTS.md §4 / §5.4：线性刺激重建解码器
# 核心设计：用充分统计量（XᵀX、Xᵀy、Σ‖x‖⁴），内存与数据量无关。
# 官方日志 6,975,446 samples × 812 columns，天真堆矩阵需要 45 GB —— 禁止。

import numpy as np


class LinearDecoder:
    """增量式线性解码器：y ≈ X b。

    observe() 只累加：
      XtX : (d, d)   ≈ 5 MB @ 812 特征
      Xty : (d,)
      x2  : Σ‖x‖²    （trace 用）
      x4  : Σ‖x‖⁴    （Ledoit-Wolf 收缩强度用）
    """

    def __init__(self, n_features):
        self.d = int(n_features)
        self.XtX = np.zeros((self.d, self.d))
        self.Xty = np.zeros(self.d)
        self.x2 = 0.0
        self.x4 = 0.0
        self.n = 0
        self.b = None

    def observe(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.shape[1] != self.d:
            raise ValueError("特征维度不符: %d != %d" % (X.shape[1], self.d))
        self.XtX += X.T @ X
        self.Xty += X.T @ y
        sq = np.einsum("ij,ij->i", X, X)  # 每行 ‖x‖²
        self.x2 += float(sq.sum())
        self.x4 += float((sq * sq).sum())
        self.n += X.shape[0]

    def solve(self, reg="lwcov", alpha=1.0):
        """求解 b。reg: 'lwcov'（默认，与官方 defaultParams.m 一致）| 'ridge' | 'none'。"""
        n = self.n
        if n <= self.d:
            raise ValueError("样本数 (%d) 不足以求解 %d 维模型" % (n, self.d))
        S = self.XtX / n
        y_target = self.Xty / n
        d = self.d

        if reg == "lwcov":
            mu = np.trace(S) / d
            delta2 = float(((S - mu * np.eye(d)) ** 2).sum() / d)
            # b̄² = (1/(n²d)) Σ_i ‖x_i x_iᵀ − S‖²_F = (x4 − ‖XtX‖²_F/n) / (n² d)
            bbar2 = max((self.x4 - float((self.XtX ** 2).sum()) / n) / (n * n * d), 0.0)
            rho = 0.0 if delta2 <= 0 else min(max(bbar2 / delta2, 0.0), 1.0)
            C = (1.0 - rho) * S + rho * mu * np.eye(d)
        elif reg == "ridge":
            mu = np.trace(S) / d
            C = S + float(alpha) * mu * np.eye(d)
        elif reg == "none":
            C = S
        else:
            raise ValueError("未知正则: %s" % reg)

        self.b = np.linalg.solve(C + 1e-12 * np.eye(d), y_target)
        return self.b

    def predict(self, X):
        if self.b is None:
            raise RuntimeError("先调用 solve()")
        return np.asarray(X, dtype=np.float64) @ self.b
