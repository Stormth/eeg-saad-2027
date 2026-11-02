# AGENTS.md §4 / §8：评测口径（改代码必须对齐）
#
# - Acc：先按被试算，再对测试集内被试等权平均。
# - L：对每个真实切换 t0，在 [t0, t0+100s] 内找 τ* 最大化 agreement；
#      要求 τ* 之后有 ≥0.5 s（64 样本 @128 Hz）连续命中新标签；
#      SDT = |τ* − t0|；ℓ = exp(−SDT/60)；被试内取中位数，再跨被试取均值。
#      未检测到 = 0 分。
# - H = 2·Acc·L/(Acc+L)。
# - 切换必须逐 trial 统计（score_subject 已按 trial 分组），不能把一个 trial
#   的末尾和下个 trial 的开头算成一次切换。

import numpy as np

SWITCH_HORIZON_S = 100.0   # 检测窗口
SWITCH_GAP_S = 0.5         # τ* 之后需要的连续命中时长
LATENCY_TAU = 60.0         # ℓ = exp(-SDT/60)


def accuracy(true, pred):
    true = np.asarray(true).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(true) == 0:
        return 0.0
    return float(np.mean(true == pred))


def _switches(x):
    """trial 内真实切换点（0-based，指向新标签的第一个样本）。"""
    x = np.asarray(x)
    if len(x) < 2:
        return []
    return [int(i) for i in np.nonzero(np.diff(x) != 0)[0] + 1]


def switch_latency(true, pred, fs):
    """单个 trial 的切换检测。

    返回 (latencies, n_switches)：latencies 是每个真实切换的 SDT（秒），
    未检测到的切换计为 None。
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    n = len(true)
    gap = max(1, int(round(SWITCH_GAP_S * fs)))
    horizon = int(round(SWITCH_HORIZON_S * fs))

    latencies = []
    for t0 in _switches(true):
        new_label = true[t0]
        end = min(n, t0 + horizon)
        best_t, best_agree = None, -1.0
        for t in range(t0, end - gap + 1):
            # τ* 之后必须有 ≥0.5 s 连续命中新标签
            if np.all(pred[t:t + gap] == new_label):
                agree = float(np.mean(pred[t:end] == true[t:end]))
                if agree > best_agree:
                    best_agree, best_t = agree, t
        if best_t is None:
            latencies.append(None)
        else:
            latencies.append((best_t - t0) / float(fs))
    return latencies, len(_switches(true))


def latency_score(latencies):
    """由一被试的切换延迟列表算 L：ℓ=exp(−SDT/60)，被试内取中位数。

    AGENTS.md §6：合成切换产生的 L/SDT 不可当真实成绩（调用方须标记 SYNTHETIC）。
    """
    det = [float(l) for l in latencies if l is not None]
    if not det:
        return 0.0
    ell = [float(np.exp(-sdt / LATENCY_TAU)) for sdt in det]
    return float(np.median(ell))


def final_score(acc, lat):
    """H = 2·A·L/(A+L)。"""
    denom = acc + lat
    if denom <= 0:
        return 0.0
    return float(2.0 * acc * lat / denom)


def score_subject(trials_true, trials_pred, fs):
    """按被试评分。

    trials_true / trials_pred : 每 trial 的逐样本 0/1 标签列表（按 trial 分组！）
    返回 dict(acc, L, sdt_median, H, n_switches, n_detected)
    """
    accs, all_lat = [], []
    n_sw = n_det = 0
    for true, pred in zip(trials_true, trials_pred):
        accs.append(accuracy(true, pred))
        lat, nsw = switch_latency(true, pred, fs)
        all_lat.extend(lat)
        n_sw += nsw
        n_det += sum(1 for l in lat if l is not None)
    acc = float(np.mean(accs)) if accs else 0.0
    lat_score = latency_score(all_lat)
    sdts = sorted(float(l) for l in all_lat if l is not None)
    sdt_median = float(np.median(sdts)) if sdts else None
    return {
        "acc": acc,
        "L": lat_score,
        "H": final_score(acc, lat_score),
        "sdt_median": sdt_median,
        "n_switches": n_sw,
        "n_detected": n_det,
    }
