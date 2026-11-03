# AGENTS.md §4：★ Config 在这里；train_on_subjects / run_subject / score_subject
#
# 设计约束（勿回退）：
# - 训练目标逐样本取（§5.3）：y = env[rows, att[rows]-1]，勿改回按 trial 取
# - 充分统计量累加（§5.4）：内存与数据量无关
# - HMM 归一化常数来自训练集（§5.6）：sigma_global / mean_global
# - expand_to_samples 用 b_k = clip(start_k + seg - la, 0, n)（§5.7），改前先跑测试

import copy
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .cnd import Subject
from .embed import time_delay_embed
from .hmm import (DEFAULT_MEAN_DIFF, DEFAULT_MEAN_GLOBAL, DEFAULT_P_SWITCH,
                  DEFAULT_SIGMA_GLOBAL, causal_hmm, normalisation_constants)
from .linear import LinearDecoder
from .metrics import accuracy, final_score, latency_score, score_subject, switch_latency
from .registry import channel_intersection


@dataclass
class Config:
    # 与官方 defaultParams.m 对齐：Leeg=[0,0.3]、lwcov、targetFs=20
    lags: Tuple[float, ...] = (0.0, 0.3)
    window: float = 1.0      # 判决窗长 (s)
    hop: float = 1.0         # 判决窗步进 (s)；hop < window 会重叠喂 HMM → 需温度缩放
    lookahead: float = 1.0   # 0=判决只标注窗之后的样本（官方保守）；=window=标注自己的窗
    reg: str = "lwcov"
    target_fs: float = 20.0  # 训练行降采样频率
    prefix: str = "pre_"     # 32 Hz 数据用 "pre_"，128 Hz 用 ""
    hmm: bool = True
    p_switch: float = DEFAULT_P_SWITCH
    mean_diff: float = DEFAULT_MEAN_DIFF
    sigma_global: float = DEFAULT_SIGMA_GLOBAL
    mean_global: float = DEFAULT_MEAN_GLOBAL
    temp: float = 1.0        # HMM 似然温度（重叠窗时 > 1）


class ChannelStandardizer:
    """按通道 z-score；统计量只从训练集累积。"""

    def __init__(self):
        self.sum_ = None
        self.sumsq = None
        self.n = 0
        self.mu = None
        self.sd = None

    def observe(self, eeg):
        if self.sum_ is None:
            self.sum_ = eeg.sum(axis=0)
            self.sumsq = (eeg ** 2).sum(axis=0)
            self.n = eeg.shape[0]
        else:
            self.sum_ += eeg.sum(axis=0)
            self.sumsq += (eeg ** 2).sum(axis=0)
            self.n += eeg.shape[0]

    def finalize(self):
        self.mu = self.sum_ / self.n
        var = self.sumsq / self.n - self.mu ** 2
        self.sd = np.sqrt(np.clip(var, 1e-12, None))

    def apply(self, eeg):
        return (eeg - self.mu) / self.sd


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def expand_to_samples(starts, decisions, n, seg, hop, lookahead_samples):
    """把窗级判决映射到逐样本标签（AGENTS.md §5.7）。

    b_k = clip(start_k + seg - la, 0, n)：判决 k 标注 [b_k, min(b_k + hop, n))。
    la=0 → 只标注窗之后的样本（严格保守）；la=seg → 判决标注自己的窗（规则上限）。
    首尾越界样本用最近判决填充。
    """
    seg = int(seg)
    hop = int(hop)
    la = int(lookahead_samples)
    n = int(n)
    labels = np.full(n, -1, dtype=int)
    if len(starts) == 0:
        labels[:] = 0
        return labels
    for k, d in enumerate(decisions):
        lo = min(max(starts[k] + seg - la, 0), n)
        hi = min(lo + hop, n)
        labels[lo:hi] = int(d)
    head = np.nonzero(labels == -1)[0]
    if len(head):
        first_b = min(max(starts[0] + seg - la, 0), n)
        labels[:first_b] = int(decisions[0])
        rest = labels == -1
        if np.any(rest):
            labels[rest] = int(decisions[-1])
    return labels


def _trial_window_corrs(pred, env, fs, window, hop):
    """单 trial：判决窗内 corr(pred, env_j)。返回 (corr, starts, seg, hop_samp)。"""
    n = len(pred)
    seg = max(1, int(round(window * fs)))
    h = max(1, int(round(hop * fs)))
    starts = list(range(0, n - seg + 1, h))
    corr = np.zeros((len(starts), env.shape[1]))
    for k, s in enumerate(starts):
        for j in range(env.shape[1]):
            corr[k, j] = pearson(pred[s:s + seg], env[s:s + seg, j])
    return corr, starts, seg, h


class SRModel:
    """训练产物：权重 + 通道 + 标准化参数 + HMM 常数 + 配置。"""

    def __init__(self, b, channels, mu, sd, lags, fs, mean_global, sigma_global,
                 config):
        self.b = np.asarray(b, dtype=np.float64)
        self.channels = list(channels)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sd = np.asarray(sd, dtype=np.float64)
        self.lags = tuple(float(l) for l in lags)
        self.fs = float(fs)
        self.mean_global = float(mean_global)
        self.sigma_global = float(sigma_global)
        self.config = dict(config)

    def save(self, path):
        np.savez(
            path, b=self.b, channels=np.array(self.channels), mu=self.mu, sd=self.sd,
            lags=np.array(self.lags), fs=self.fs,
            mean_global=self.mean_global, sigma_global=self.sigma_global,
            config=np.array([repr(self.config)]),
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        return cls(z["b"], [str(c) for c in z["channels"]], z["mu"], z["sd"],
                   z["lags"], float(z["fs"]), float(z["mean_global"]),
                   float(z["sigma_global"]), eval(str(z["config"][0])))


def train_on_subjects(subjects, cfg):
    """跨数据集合并训练线性 SR 解码器（AGENTS.md §9 P0-3）。"""
    if not subjects:
        raise ValueError("没有训练被试")
    channels = channel_intersection(subjects)
    if not channels:
        raise ValueError("训练被试之间没有通道交集")
    subjects = [copy.deepcopy(s).select_channels(channels) for s in subjects]

    std = ChannelStandardizer()
    for s in subjects:
        for e in s.eeg:
            std.observe(e)
    std.finalize()

    n_feat = len(channels) * len(cfg.lags)
    dec = LinearDecoder(n_feat)
    step = max(1, int(round(subjects[0].fs / cfg.target_fs)))
    for s in subjects:
        for eeg, env, att in zip(s.eeg, s.env, s.att):
            if att is None:
                raise ValueError("训练被试 %s 缺少 attendedStream" % s.source)
            eeg_s = std.apply(eeg)
            D, t0, t1, _ = time_delay_embed(eeg_s, cfg.lags, s.fs)
            rows = np.arange(0, t1, step)  # §5.3：逐样本目标，按 target_fs 降采样
            att_seg = np.asarray(att).reshape(-1)[rows]
            y = env[rows, att_seg - 1]     # 勿改回 env[:, int(att[0])-1]
            dec.observe(D[rows], y)
    b = dec.solve(cfg.reg)

    # §5.6：归一化常数来自训练集 corr（在训练数据上回放判决窗）
    all_corr, all_att = [], []
    for s in subjects:
        for eeg, env, att in zip(s.eeg, s.env, s.att):
            eeg_s = std.apply(eeg)
            D, t0, t1, _ = time_delay_embed(eeg_s, cfg.lags, s.fs)
            pred = D @ b
            corr, starts, seg, h = _trial_window_corrs(pred, env, s.fs,
                                                       cfg.window, cfg.hop)
            att_s = np.asarray(att).reshape(-1)
            att_w = []
            for st in starts:
                vals, cnt = np.unique(att_s[st:st + seg], return_counts=True)
                att_w.append(int(vals[np.argmax(cnt)]) - 1)  # → 0-based
            all_corr.append(corr)
            all_att.extend(att_w)
    all_corr = np.concatenate(all_corr, axis=0)
    mean_g, sigma_g = normalisation_constants(all_corr, np.array(all_att))

    model = SRModel(b, channels, std.mu, std.sd, cfg.lags, subjects[0].fs,
                    mean_g, sigma_g,
                    {"lags": list(cfg.lags), "window": cfg.window, "hop": cfg.hop,
                     "lookahead": cfg.lookahead, "reg": cfg.reg,
                     "target_fs": cfg.target_fs, "hmm": cfg.hmm,
                     "p_switch": cfg.p_switch, "mean_diff": cfg.mean_diff,
                     "sigma_global": cfg.sigma_global, "mean_global": cfg.mean_global,
                     "temp": cfg.temp})
    return model


def _make_model(b, channels, mu, sd, fs, mean_g, sigma_g, cfg):
    return SRModel(b, channels, mu, sd, cfg.lags, fs, mean_g, sigma_g,
                   {"lags": list(cfg.lags), "window": cfg.window, "hop": cfg.hop,
                    "lookahead": cfg.lookahead, "reg": cfg.reg,
                    "target_fs": cfg.target_fs, "hmm": cfg.hmm,
                    "p_switch": cfg.p_switch, "mean_diff": cfg.mean_diff,
                    "sigma_global": cfg.sigma_global, "mean_global": cfg.mean_global,
                    "temp": cfg.temp})


def _iter_spec(specs, limit_per_dataset=None, loader=None):
    """逐个产出 (key, Subject)。specs: [(data_root, key, prefix, subjects|None), ...]

    默认用 cnd.load_subject 从磁盘按需加载 —— 每次只有一个被试在内存里。
    内存与数据总量无关（P0 五库合并训练必需）。
    """
    from . import cnd

    for root, key, prefix, subs in specs:
        ids = subs if subs else cnd.list_subjects(root, key, prefix)
        if limit_per_dataset:
            ids = list(ids)[:limit_per_dataset]
        for j in ids:
            if loader is None:
                yield key, cnd.load_subject(root, key, j, prefix)
            else:
                yield key, loader(root, key, j, prefix)


def train_streaming(specs, cfg, limit_per_dataset=None, const_subjects=3,
                    loader=None, log=print):
    """流式合并训练（AGENTS.md §9 P0-3 的正确姿势）。

    specs = [("data", "DTU", "pre_", None), ("data", "KUL", "pre_", None), ...]

    分三遍扫描磁盘，内存峰值 = 单个被试：
      1) 通道交集
      2) 通道 z-score 统计量
      3) 累加充分统计量 → 求解 b
    第 4 遍只在前 const_subjects 个被试上估计 HMM 归一化常数（§5.6）。
    """
    # pass 1：通道交集（保持第一个被试的通道顺序）
    channels = None
    n_subjects = 0
    for _, s in _iter_spec(specs, limit_per_dataset, loader):
        if channels is None:
            channels = list(s.channels)
        else:
            allowed = set(s.channels)
            channels = [c for c in channels if c in allowed]
        n_subjects += 1
        del s
    if not channels:
        raise ValueError("训练被试之间没有通道交集（检查数据集通道命名）")
    log("通道交集: %d 通道（%d 被试）" % (len(channels), n_subjects))

    # pass 2：标准化统计量
    std = ChannelStandardizer()
    for _, s in _iter_spec(specs, limit_per_dataset, loader):
        s.select_channels(channels)
        for e in s.eeg:
            std.observe(e)
        del s
    std.finalize()

    # pass 3：充分统计量
    dec = LinearDecoder(len(channels) * len(cfg.lags))
    fs0 = None
    for _, s in _iter_spec(specs, limit_per_dataset, loader):
        s.select_channels(channels)
        fs0 = fs0 or s.fs
        step = max(1, int(round(s.fs / cfg.target_fs)))
        for eeg, env, att in zip(s.eeg, s.env, s.att):
            if att is None:
                continue
            eeg_s = std.apply(eeg)
            D, t0, t1, _ = time_delay_embed(eeg_s, cfg.lags, s.fs)
            rows = np.arange(0, t1, step)
            att_seg = np.asarray(att).reshape(-1)[rows]
            dec.observe(D[rows], env[rows, att_seg - 1])
        del s
    b = dec.solve(cfg.reg)

    # pass 4：HMM 归一化常数（只用前 const_subjects 个被试，§5.6 必须来自训练集）
    all_corr, all_att = [], []
    for i, (_, s) in enumerate(_iter_spec(specs, const_subjects, loader)):
        s.select_channels(channels)
        eeg_std = [(e - std.mu) / std.sd for e in s.eeg]
        for eeg_s, env, att in zip(eeg_std, s.env, s.att):
            if att is None:
                continue
            D, t0, t1, _ = time_delay_embed(eeg_s, cfg.lags, s.fs)
            pred = D @ b
            corr, starts, seg, _h = _trial_window_corrs(pred, env, s.fs,
                                                        cfg.window, cfg.hop)
            att_s = np.asarray(att).reshape(-1)
            for st in starts:
                vals, cnt = np.unique(att_s[st:st + seg], return_counts=True)
                all_att.append(int(vals[np.argmax(cnt)]) - 1)
            all_corr.append(corr)
        del s
    if all_corr:
        mean_g, sigma_g = normalisation_constants(np.concatenate(all_corr, axis=0),
                                                  np.array(all_att))
    else:
        mean_g, sigma_g = cfg.mean_global, cfg.sigma_global

    return _make_model(b, channels, std.mu, std.sd, fs0, mean_g, sigma_g, cfg)


def run_subject(model, subject, cfg=None, hmm=None):
    """对单个被试做逐样本判决。

    返回 list[dict]，每 trial：
      true    : (n,) 0/1 真标签（att 为 None 时也是 None，用于盲测）
      pred    : (n,) 0/1 预测标签
      corr    : (n_windows, 2)
      decisions, starts, fs
    """
    if cfg is None:
        cfg = Config(**{k: v for k, v in model.config.items()})
    if hmm is None:
        hmm = bool(cfg.hmm)

    s = copy.deepcopy(subject).select_channels(model.channels)
    results = []
    dt_s = cfg.hop
    for eeg, env, att in zip(s.eeg, s.env, s.att):
        eeg_s = (eeg - model.mu) / model.sd
        D, t0, t1, _ = time_delay_embed(eeg_s, model.lags, s.fs)
        pred_sig = D @ model.b
        corr, starts, seg, h = _trial_window_corrs(pred_sig, env, s.fs,
                                                   cfg.window, cfg.hop)
        if len(starts) == 0:
            dec = np.zeros(1, dtype=int)
            starts = [0]
        elif hmm:
            dec, _ = causal_hmm(corr, model.mean_global, model.sigma_global,
                                dt_s, p_switch=cfg.p_switch, temp=cfg.temp)
        else:
            dec = np.argmax(corr, axis=1).astype(int)
        n = eeg.shape[0]
        labels = expand_to_samples(starts, dec, n, seg, h,
                                   round(cfg.lookahead * s.fs))
        true = None if att is None else (np.asarray(att).reshape(-1) - 1)
        results.append({"true": true, "pred": labels, "corr": corr,
                        "decisions": dec, "starts": starts, "fs": s.fs})
    return results


def score_subject(subject, results):
    """按 AGENTS.md §8 评分（先按被试，逐 trial 统计切换）。"""
    trues = [r["true"] for r in results if r["true"] is not None]
    preds = [r["pred"] for r in results if r["true"] is not None]
    if not trues:
        return None
    return score_subject_generic(trues, preds, subject.fs)


def score_subject_generic(trials_true, trials_pred, fs):
    return score_subject(trials_true, trials_pred, fs)
