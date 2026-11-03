# AGENTS.md §7：主入口 —— 训练 / 评测 / 盲测预测
#
# 1) 单被试快速验证：
#   python scripts/run_linear.py --data-root data --train DTU --test DTU \
#       --test-subjects 1 --out outputs/check --synthetic-switches 2
# 2) 正式 4 库训练 + AVGC 验证：
#   python scripts/run_linear.py --data-root data \
#       --train DTU KUL UHD USTC --test AVGC --out outputs/baseline
# 3) 盲测提交（测试阶段开放后）：
#   python scripts/run_linear.py --data-root data \
#       --test-unknown data/BLIND/dataCND --load outputs/baseline/model.npz \
#       --submission outputs/submission.csv

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from eegsaad import cnd, metrics, pipeline, registry, submit  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="ICASSP 2027 GC-11 线性 SR + 因果 HMM 基线")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train", nargs="+", default=None,
                   help="训练数据集别名，如 DTU KUL UHD USTC")
    p.add_argument("--test", nargs="+", default=None, help="验证数据集别名（如 AVGC）")
    p.add_argument("--test-subjects", nargs="+", type=int, default=None,
                   help="只测这些被试编号（默认全部）")
    p.add_argument("--out", default="outputs/run", help="输出目录")
    p.add_argument("--load", default=None, help="加载已有 model.npz 跳过训练")
    p.add_argument("--test-unknown", default=None,
                   help="盲测 dataCND 目录（无标签），写提交文件")
    p.add_argument("--submission", default=None, help="提交 CSV 输出路径")
    p.add_argument("--synthetic-switches", type=int, default=0,
                   help="给测试被试注入合成切换（L/SDT 不可当真实成绩）")
    # 与官方 defaultParams.m 对齐的参数
    p.add_argument("--lags", nargs="+", type=float, default=[0.0, 0.3])
    p.add_argument("--window", type=float, default=1.0)
    p.add_argument("--hop", type=float, default=1.0)
    p.add_argument("--lookahead", type=float, default=1.0)
    p.add_argument("--reg", default="lwcov", choices=["lwcov", "ridge", "none"])
    p.add_argument("--target-fs", type=float, default=20.0)
    p.add_argument("--prefix", default="pre_",
                   help='32 Hz 数据用 "pre_"，128 Hz 数据用 ""')
    p.add_argument("--no-hmm", action="store_true")
    p.add_argument("--p-switch", type=float, default=0.001)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--no-stream", action="store_true",
                   help="关掉流式训练（会把全部被试读进内存，仅小数据用）")
    p.add_argument("--limit-per-dataset", type=int, default=None,
                   help="每个数据集只用前 N 个被试（调试用）")
    return p


def evaluate_dataset(model, key, data_root, prefix, subjects_filter, cfg,
                     synthetic_switches, out_dir):
    print("== 评测 %s ==" % key)
    if subjects_filter:
        subjects = [cnd.load_subject(data_root, key, s, prefix)
                    for s in subjects_filter]
    else:
        subjects = cnd.load_dataset(data_root, key, prefix)
    per_subject = {}
    for s in subjects:
        if synthetic_switches > 0:
            # AGENTS.md §5.8 / §10：合成切换同时交换 env 列与标签，Acc 不变、
            # 只有时间结构变；由此得到的 L/SDT 不能当真实成绩
            from eegsaad.synthetic import inject_synthetic_switches
            s = inject_synthetic_switches(s, synthetic_switches, seed=0)
        results = pipeline.run_subject(model, s, cfg)
        sc = pipeline.score_subject(s, results)
        if sc is None:
            print("  %s subj?: 无标签，跳过评分" % s.source)
            continue
        tag = "subj%s" % (s.source.split("subj")[-1])
        per_subject[tag] = sc
        per_subject[tag]["synthetic"] = bool(s.synthetic)
        print("  %-12s Acc=%.3f  L=%.3f  H=%.3f  switches=%d/%d%s" % (
            tag, sc["acc"], sc["L"], sc["H"], sc["n_detected"], sc["n_switches"],
            "  [SYNTHETIC-SWITCHES]" if s.synthetic else ""))
    if not per_subject:
        return None
    agg = {
        "acc": float(np.mean([v["acc"] for v in per_subject.values()])),
        "L": float(np.mean([v["L"] for v in per_subject.values()])),
    }
    agg["H"] = metrics.final_score(agg["acc"], agg["L"])
    print("  %-12s Acc=%.3f  L=%.3f  H=%.3f" % ("MEAN", agg["acc"], agg["L"], agg["H"]))
    return {"per_subject": per_subject, "mean": agg}


def predict_unknown(model, unknown_dir, cfg, submission_path):
    """盲测目录：读所有 dataSub*.mat，写提交 CSV。"""
    import glob
    import re

    paths = sorted(glob.glob(os.path.join(unknown_dir, "*dataSub*.mat")))
    if not paths:
        raise FileNotFoundError("%s 里没有 dataSub*.mat" % unknown_dir)
    trial_labels, source_fs = [], model.fs
    key = "BLIND"
    registry.ALIASES.setdefault(key, os.path.basename(
        os.path.dirname(os.path.normpath(unknown_dir))))
    os.makedirs(os.path.dirname(os.path.abspath(submission_path)), exist_ok=True)
    n_rows = 0
    for p in paths:
        m = re.search(r"dataSub(\d+)", os.path.basename(p))
        subj = int(m.group(1)) if m else len(trial_labels) + 1
        s = cnd.load_subject(os.path.dirname(os.path.dirname(unknown_dir)),
                             key, subj, prefix=cfg.prefix)
        results = pipeline.run_subject(model, s, cfg)
        for r in results:
            trial_labels.append(r["pred"])
            source_fs = r["fs"]
        print("  dataSub%d: %d trials" % (subj, len(results)))
    n_rows = submit.write_submission(submission_path, trial_labels, source_fs)
    print("提交文件已写: %s (%d 行)" % (submission_path, n_rows))
    return submission_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = pipeline.Config(
        lags=tuple(args.lags), window=args.window, hop=args.hop,
        lookahead=args.lookahead, reg=args.reg, target_fs=args.target_fs,
        prefix=args.prefix, hmm=not args.no_hmm, p_switch=args.p_switch,
        temp=args.temp,
    )
    os.makedirs(args.out, exist_ok=True)

    model = None
    if args.load:
        model = pipeline.SRModel.load(args.load)
        print("已加载模型: %s" % args.load)
    elif args.train:
        if args.no_stream:
            subjects = []
            for key in args.train:
                subs = cnd.load_dataset(args.data_root, key, cfg.prefix)
                print("载入 %s: %d 被试" % (key, len(subs)))
                subjects.extend(subs)
            print("合并训练中（%d 被试, %d 通道交集）..." % (
                len(subjects), len(registry.channel_intersection(subjects))))
            model = pipeline.train_on_subjects(subjects, cfg)
        else:
            # 流式：三遍扫描磁盘，内存峰值 = 单被试（5 库合并必需）
            specs = [(args.data_root, k, cfg.prefix, None) for k in args.train]
            print("流式合并训练: %s（每数据集上限 %s）" % (
                " ".join(args.train), args.limit_per_dataset or "无"))
            model = pipeline.train_streaming(
                specs, cfg, limit_per_dataset=args.limit_per_dataset)
        model_path = os.path.join(args.out, "model.npz")
        model.save(model_path)
        print("模型已存: %s（%d 通道, HMM 常数 mean=%.3f sigma=%.3f）" % (
            model_path, len(model.channels), model.mean_global, model.sigma_global))
    else:
        raise SystemExit("需要 --train 或 --load")

    report = {"config": model.config,
              "hmm_constants": {"mean_global": model.mean_global,
                                "sigma_global": model.sigma_global},
              "datasets": {}}
    if args.test:
        for key in args.test:
            rep = evaluate_dataset(model, key, args.data_root, cfg.prefix,
                                   args.test_subjects, cfg,
                                   args.synthetic_switches, args.out)
            if rep is not None:
                report["datasets"][key] = rep

    report_path = os.path.join(args.out, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("报告已写: %s" % report_path)

    if args.test_unknown:
        if not args.submission:
            raise SystemExit("--test-unknown 需要同时给 --submission")
        predict_unknown(model, args.test_unknown, cfg, args.submission)
    return 0


if __name__ == "__main__":
    sys.exit(main())
