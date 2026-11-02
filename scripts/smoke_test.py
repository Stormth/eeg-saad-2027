# AGENTS.md §7：10 秒跑通全流程（合成数据，无需真实数据）

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from eegsaad import metrics, pipeline, submit, synthetic  # noqa: E402


def main():
    print("[1/4] 生成合成数据集 ...")
    subjects = synthetic.make_dataset(n_subjects=2, n_trials=3, fs=32.0,
                                      n_samples=9600, n_channels=8, seed=0)
    cfg = pipeline.Config(lags=(0.0, 0.1), window=1.0, hop=1.0,
                          lookahead=1.0, reg="lwcov", target_fs=20.0)

    print("[2/4] 训练（被试 1）→ 评测（被试 2）...")
    model = pipeline.train_on_subjects(subjects[:1], cfg)
    results = pipeline.run_subject(model, subjects[1], cfg)
    trues = [r["true"] for r in results]
    preds = [r["pred"] for r in results]
    acc = metrics.accuracy(np.concatenate(trues), np.concatenate(preds))
    print("  纯线性 SR Acc = %.3f" % acc)
    if acc < 0.7:
        raise SystemExit("FAIL: 合成数据 Acc 过低 (%.3f)，管线有问题" % acc)

    print("[3/4] 因果 HMM ...")
    results_hmm = pipeline.run_subject(model, subjects[1], cfg, hmm=True)
    acc_hmm = metrics.accuracy(
        np.concatenate([r["true"] for r in results_hmm]),
        np.concatenate([r["pred"] for r in results_hmm]))
    print("  + 因果 HMM Acc = %.3f" % acc_hmm)
    if acc_hmm < acc - 0.02:
        raise SystemExit("FAIL: HMM 明显劣化 (%.3f < %.3f)" % (acc_hmm, acc))

    print("[4/4] 128 Hz 提交读写往返 ...")
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "outputs", "_smoke_submission.csv")
    os.makedirs(os.path.dirname(os.path.abspath(tmp)), exist_ok=True)
    submit.write_submission(tmp, preds, source_fs=results[0]["fs"])
    back = submit.read_submission(tmp)
    assert len(back) == len(preds), "trial 数不符"
    for a, b in zip(back, preds):
        b2 = submit.zoh_upsample(b, results[0]["fs"])
        assert len(a) == len(b2) and np.array_equal(a, b2), "往返内容不符"
    os.remove(tmp)

    print("全部通过")


if __name__ == "__main__":
    main()
