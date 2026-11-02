# AGENTS.md §4 / §6
# 读取 extended CND .mat 文件 → Subject 对象。
#
# 已知坑（回归测试覆盖，勿回退）：
# 1. scipy.io.loadmat 把 struct 包成 (1,1) object 数组      -> _unwrap()
# 2. stim.names 元素是 ['envelope_stream1'] 而非字符串       -> _scalar_str()
# 3. stim.data 是 {F×N} cell，np.array(list).reshape 会广播  -> 直接二维索引 cell[i, t]

import os
import re
import warnings

import numpy as np
from scipy.io import loadmat


def _unwrap(x):
    """递归剥掉 loadmat 产生的 (1,1) object 数组包装。"""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x[0, 0] if x.ndim == 2 else x.reshape(-1)[0]
    return x


def _scalar_str(x):
    """把 cell/数组层层剥开，取出一个字符串。"""
    x = _unwrap(x)
    for _ in range(8):
        if isinstance(x, str):
            return x
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="replace")
        if isinstance(x, (list, tuple, np.ndarray)):
            if isinstance(x, np.ndarray):
                if x.size == 0:
                    return ""
                x = x.reshape(-1)[0]
            else:
                x = x[0]
            x = _unwrap(x)
        else:
            return str(x)
    return str(x)


def _fields(rec):
    """返回 struct 的字段名列表（不是 struct 则返回 None）。"""
    rec = _unwrap(rec)
    if isinstance(rec, np.ndarray) and rec.dtype.names:
        return list(rec.dtype.names)
    return None


def _get(rec, name):
    """从 (1,1) struct 数组取字段。"""
    return _unwrap(rec[0, 0][name])


def _find_field(names, *keywords):
    """字段名模糊匹配（大小写不敏感）。"""
    if not names:
        return None
    for n in names:
        ln = n.lower()
        if all(k in ln for k in keywords):
            return n
    return None


class Subject:
    """一个被试的全部 trial。

    属性
    ----
    fs : float                    采样率 (Hz)
    eeg : list[np.ndarray]        每 trial (n_samples, n_channels)
    env : list[np.ndarray]        每 trial (n_samples, 2)，两路语音包络
    att : list[np.ndarray]        每 trial (n_samples,)，1/2 表示在听哪路（保持 1-based 原始值）
    az : list or None             每 trial 方位角（缺失为 None）
    channels : list[str]          通道名
    """

    def __init__(self, fs, eeg, env, att, az, channels, source="", synthetic=False):
        self.fs = float(fs)
        self.eeg = eeg
        self.env = env
        self.att = att
        self.az = az
        self.channels = list(channels)
        self.source = source
        self.synthetic = bool(synthetic)

    def __len__(self):
        return len(self.eeg)

    def select_channels(self, target):
        """对齐到目标通道列表（用于跨数据集合并训练）。

        缺失通道补零并 warning；多余通道丢弃。
        AGENTS.md §9 P0-2：此方法从未在真数据上验证，合并训练时必须先跑通。
        """
        target = list(target)
        idx = {c: i for i, c in enumerate(self.channels)}
        missing = [c for c in target if c not in idx]
        if missing:
            warnings.warn(
                "%s: %d/%d 通道缺失，补零: %s%s"
                % (self.source, len(missing), len(target), missing[:5],
                   "..." if len(missing) > 5 else "")
            )
        new_eeg = []
        for e in self.eeg:
            out = np.zeros((e.shape[0], len(target)), dtype=np.float64)
            for j, c in enumerate(target):
                if c in idx:
                    out[:, j] = e[:, idx[c]]
            new_eeg.append(out)
        self.eeg = new_eeg
        self.channels = target
        return self


def _resample_lin(x, n_target):
    """线性插值把 (n,) 重采样到 n_target 点。"""
    n = len(x)
    if n == n_target:
        return x
    if n == 0:
        return np.zeros(n_target)
    return np.interp(np.linspace(0, n - 1, n_target), np.arange(n), x)


def _resample_nearest(x, n_target):
    n = len(x)
    if n == n_target:
        return x
    if n == 0:
        return np.zeros(n_target, dtype=np.float64)
    idx = np.clip(np.round(np.linspace(0, n - 1, n_target)).astype(int), 0, n - 1)
    return x[idx]


def _load_struct(path, required):
    """读 .mat，自动发现含全部 required 字段的顶层 struct。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    mat = loadmat(path, squeeze_me=False, struct_as_record=True)
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        names = _fields(v)
        if names and all(_find_field(names, r) for r in required):
            return v
    # 没找到：报错并列出可用字段
    found = {}
    for k, v in mat.items():
        if not k.startswith("__"):
            found[k] = _fields(v)
    raise KeyError(
        "%s: 找不到含字段 %s 的 struct；顶层内容: %s" % (path, required, found)
    )


def _channel_names(rec, names):
    ch_field = _find_field(names, "channel")
    if ch_field is None:
        return None
    raw = _get(rec, ch_field)
    raw = np.atleast_2d(np.asarray(raw, dtype=object))
    out = []
    for cell in raw.reshape(-1):
        s = _scalar_str(cell)
        if s:
            out.append(s)
    return out


def load_subject(data_root, key, subject, prefix="pre_"):
    """读单个被试：dataCND/{prefix}dataSub{j}.mat + {prefix}dataStim{j}.mat。

    att/包络/EEG 长度不一致时自动重采样对齐到 EEG 长度。
    """
    from .registry import folder_for

    folder = folder_for(data_root, key)
    eeg_path = os.path.join(folder, "%sdataSub%d.mat" % (prefix, subject))
    stim_path = os.path.join(folder, "%sdataStim%d.mat" % (prefix, subject))

    eeg_rec = _load_struct(eeg_path, ["fs", "data"])
    stim_rec = _load_struct(stim_path, ["names", "data"])

    eeg_names = _fields(eeg_rec)
    stim_names = _fields(stim_rec)

    fs = float(np.asarray(_get(eeg_rec, _find_field(eeg_names, "fs"))).reshape(-1)[0])

    channels = _channel_names(eeg_rec, eeg_names)
    raw = np.asarray(_get(eeg_rec, "data"), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    # 方向判别：优先用通道数匹配，否则长轴当样本轴
    if channels is not None and raw.ndim == 2:
        if raw.shape[1] == len(channels):
            pass
        elif raw.shape[0] == len(channels):
            raw = raw.T
    elif raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
        raw = raw.T  # (n, C) 通常 n >> C

    # attendedStream：在 EEG 或 Stim 两个 struct 里模糊查找（AGENTS.md §5.3：逐样本 1/2）
    att_field = _find_field(eeg_names, "attend") or _find_field(stim_names, "attend")
    att_raw = None
    if att_field is not None:
        src = eeg_rec if _find_field(eeg_names, "attend") else stim_rec
        att_raw = np.asarray(_get(src, att_field), dtype=np.float64).reshape(-1)

    # 包络：stim.data 是 {nTrials × nStreams} cell（坑 #3：直接二维索引）
    names_raw = _get(stim_rec, _find_field(stim_names, "names"))
    names_raw = np.atleast_2d(np.asarray(names_raw, dtype=object))
    stream_names = [_scalar_str(v) for v in names_raw.reshape(-1)]
    order = []
    for want in (1, 2):
        hit = None
        for i, s in enumerate(stream_names):
            m = re.search(r"envelope[_\-]?stream(\d+)", s)
            if m and int(m.group(1)) == want:
                hit = i
                break
        order.append(hit if hit is not None else want - 1)

    data_raw = _get(stim_rec, "data")
    data_cell = np.atleast_2d(np.asarray(data_raw, dtype=object))
    n_trials, _ = data_cell.shape

    stim_fs = None
    stim_fs_field = _find_field(stim_names, "fs")
    if stim_fs_field is not None:
        try:
            stim_fs = float(np.asarray(_get(stim_rec, stim_fs_field)).reshape(-1)[0])
        except Exception:
            stim_fs = None
    if stim_fs is None:
        stim_fs = fs

    eeg_trials, env_trials, att_trials, az_trials = [], [], [], []
    for i in range(n_trials):
        env_i = []
        for t in range(2):
            cell = data_cell[i, order[t]]
            col = np.asarray(cell, dtype=np.float64).reshape(-1)
            env_i.append(col)
        env_i = np.stack(env_i, axis=1)  # (n_env, 2)

        n_env = env_i.shape[0]
        att_i = att_raw if att_raw is None else np.asarray(att_raw, dtype=np.float64)
        if att_i is not None and len(att_i) == n_env and n_env != raw.shape[0]:
            att_i = _resample_nearest(att_i, raw.shape[0])
        if env_i.shape[0] != raw.shape[0]:
            env_i = np.stack(
                [_resample_lin(env_i[:, 0], raw.shape[0]),
                 _resample_lin(env_i[:, 1], raw.shape[0])],
                axis=1,
            )
        if att_i is not None and len(att_i) != raw.shape[0]:
            att_i = _resample_nearest(att_i, raw.shape[0])

        if stim_fs and abs(stim_fs - fs) > 1e-6 and env_i.shape[0] * fs // stim_fs != raw.shape[0]:
            n_target = int(round(env_i.shape[0] * fs / stim_fs))
            env_i = np.stack(
                [_resample_lin(env_i[:, 0], n_target),
                 _resample_lin(env_i[:, 1], n_target)],
                axis=1,
            )

        eeg_trials.append(raw)
        env_trials.append(env_i)
        att_trials.append(None if att_i is None else att_i.astype(int))

        az_i = None
        az_field = _find_field(eeg_names, "az") or _find_field(stim_names, "az")
        if az_field is not None:
            try:
                src = eeg_rec if _find_field(eeg_names, "az") else stim_rec
                az_i = np.asarray(_get(src, az_field), dtype=np.float64)
                if az_i.ndim == 2 and az_i.shape[0] == n_trials:
                    az_i = az_i[i]
                az_i = az_i.reshape(-1)[:2]
            except Exception:
                az_i = None
        az_trials.append(az_i)

    return Subject(fs, eeg_trials, env_trials, att_trials, az_trials,
                   channels if channels is not None else [],
                   source="%s/%s subj%d" % (key, folder, subject))


def list_subjects(data_root, key, prefix="pre_"):
    """扫描 dataCND 里实际存在的被试编号。"""
    from .registry import folder_for

    folder = folder_for(data_root, key)
    subs = set()
    if not os.path.isdir(folder):
        return sorted(subs)
    import glob as _glob

    for p in _glob.glob(os.path.join(folder, "%sdataSub*.mat" % prefix)):
        m = re.search(r"dataSub(\d+)\.mat$", os.path.basename(p))
        if m:
            subs.add(int(m.group(1)))
    return sorted(subs)


def load_dataset(data_root, key, prefix="pre_", subjects=None):
    """读整个数据集（默认所有被试），返回 Subject 列表。"""
    if subjects is None:
        subjects = list_subjects(data_root, key, prefix)
    return [load_subject(data_root, key, s, prefix) for s in subjects]
