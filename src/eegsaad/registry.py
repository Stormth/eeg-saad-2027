# AGENTS.md §4：数据集别名 → dataCND 路径

import os

ALIASES = {
    "DTU": "DTU2018",
    "KUL": "KUL2016",
    "AVGC": "AVGCAAD2024",
    "UHD": "KUL_AAD_UltraHD2021",
    "USTC": "USTC2026",
}

# AGENTS.md §1 数据表（供 describe 参考）
DATASET_INFO = {
    "DTU":   {"channels": 64,  "language": "丹麦语", "subjects": 18, "gaze_controlled": False},
    "KUL":   {"channels": 64,  "language": "荷兰语", "subjects": 16, "gaze_controlled": False},
    "AVGC":  {"channels": 64,  "language": "荷兰语", "subjects": 13, "gaze_controlled": True},
    "UHD":   {"channels": 255, "language": "荷兰语", "subjects": 30, "gaze_controlled": False},
    "USTC":  {"channels": 64,  "language": "中文",   "subjects": 18, "gaze_controlled": False},
}


def folder_for(data_root, key):
    """数据集别名 → dataCND 目录绝对/相对路径。"""
    if key not in ALIASES:
        raise KeyError("未知数据集别名: %s（可选: %s）" % (key, ", ".join(ALIASES)))
    return os.path.join(data_root, ALIASES[key], "dataCND")


def describe(data_root):
    """列出 5 个数据集的存在状态与被试数（AGENTS.md §9 P0-1 的验收命令）。"""
    import re
    import glob

    out = {}
    for key, folder_name in ALIASES.items():
        folder = os.path.join(data_root, folder_name, "dataCND")
        n_sub = 0
        if os.path.isdir(folder):
            for p in glob.glob(os.path.join(folder, "*dataSub*.mat")):
                m = re.search(r"dataSub(\d+)\.mat$", os.path.basename(p))
                if m:
                    n_sub = max(n_sub, int(m.group(1)))
        out[key] = {
            "folder": folder,
            "exists": os.path.isdir(folder),
            "n_subjects": n_sub,
            "info": DATASET_INFO[key],
        }
    return out


def channel_intersection(subjects):
    """多个 Subject 的通道交集（保持第一个被试的通道顺序）。"""
    if not subjects:
        return []
    common = list(subjects[0].channels)
    for s in subjects[1:]:
        allowed = set(s.channels)
        common = [c for c in common if c in allowed]
    return common
