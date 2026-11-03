# AGENTS.md §4 / §6：8 个自检，纯 python 可跑（无需 pytest，无需真实数据）。
# 改动 cnd.py / pipeline.py 后必须跑：python tests/test_all.py → 8/8 passed

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from eegsaad import cnd, embed, linear, metrics, pipeline, submit, synthetic  # noqa: E402
from eegsaad.hmm import causal_hmm, normalisation_constants  # noqa: E402

RESULTS = []


def check(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


@check("test_embed_design")
def test_embed():
    """时间延迟嵌入：形状、t0/t1、正 lag = 未来采样。"""
    fs = 32.0
    n, c = 1000, 4
    eeg = np.arange(n * c, dtype=float).reshape(n, c)
    D, t0, t1, li = embed.time_delay_embed(eeg, [0.0, 0.5], fs)
    assert li == [0, 16], li
    assert D.shape == (n - 16, 8), D.shape
    assert t0 == 0 and t1 == n - 16
    # 行 i 的后半 = eeg[i+16]（未来 0.5 s）
    assert np.array_equal(D[3, 4:], eeg[3 + 16])
    assert np.array_equal(D[3, :4], eeg[3])
    try:
        embed.time_delay_embed(eeg, [-0.1], fs)
        raise AssertionError("负 lag 应被拒绝（因果性）")
    except ValueError:
        pass


@check("test_linear_recovery")
def test_linear():
    """充分统计量解码器：已知权重恢复（AGENTS.md §2：权重恢复测试）。"""
    rng = np.random.default_rng(1)
    n, d = 20000, 12
    X = rng.standard_normal((n, d))
    b_true = rng.standard_normal(d)
    y = X @ b_true + 0.01 * rng.standard_normal(n)
    dec = linear.LinearDecoder(d)
    # 分 20 块喂入，验证增量累加正确
    for i in range(0, n, n // 20):
        dec.observe(X[i:i + n // 20], y[i:i + n // 20])
    b = dec.solve("lwcov")
    pred = X @ b
    r = np.corrcoef(pred, X @ b_true)[0, 1]
    assert r > 0.999, "权重恢复失败 corr=%.4f" % r
    # 两次 observe 顺序无关
    d2 = linear.LinearDecoder(d)
    d2.observe(X[:n // 2], y[:n // 2])
    d2.observe(X[n // 2:], y[n // 2:])
    assert np.allclose(d2.solve("lwcov"), b, atol=1e-8)


@check("test_metrics_perfect")
def test_metrics_perfect():
    """完美预测：Acc=1, L=1（AGENTS.md §2 验收）。"""
    fs = 128.0
    true = np.ones(25600, dtype=int)
    true[12800:] = 0  # 一次切换
    sc = metrics.score_subject([true], [true.copy()], fs)
    assert sc["acc"] == 1.0
    assert abs(sc["L"] - 1.0) < 1e-9, sc["L"]
    assert abs(sc["H"] - 1.0) < 1e-9
    assert sc["sdt_median"] == 0.0


@check("test_metrics_never_switch")
def test_metrics_never_switch():
    """从不切换的预测：检测不到切换 → L=0（AGENTS.md §2 验收）。"""
    fs = 128.0
    true = np.ones(25600, dtype=int)
    true[12800:] = 0
    const = np.ones(25600, dtype=int)
    sc = metrics.score_subject([true], [const], fs)
    assert sc["acc"] == 0.5
    assert sc["L"] == 0.0, sc["L"]
    assert sc["H"] == 0.0
    # 切换不跨 trial 统计（§8）：两个 trial 分开传 → 各 1 次切换；
    # 若天真拼接成一条序列会出现 3 次跳变，score_subject 必须按 trial 分组
    a = np.array([1, 1, 0, 0])
    lat, nsw = metrics.switch_latency(a, a, 32.0)
    assert nsw == 1
    sc2 = metrics.score_subject([a, a], [a, a], 32.0)
    assert sc2["n_switches"] == 2, sc2


@check("test_hmm_causal")
def test_hmm_causal():
    """HMM：干净输入下判决正确；前向滤波不含未来（前缀一致性）。"""
    rng = np.random.default_rng(2)
    n = 400
    mean_g, sigma_g = 0.2, 0.29
    state = np.zeros(n, dtype=int)
    state[200:] = 1
    d = np.where(state == 0, mean_g, -mean_g) + 0.05 * rng.standard_normal(n)
    z = np.zeros((n, 2))
    z[:, 0] = d / 2
    z[:, 1] = -d / 2
    dec, posts = causal_hmm(z, mean_g, sigma_g, dt=1.0)
    assert np.mean(dec == state) > 0.95, np.mean(dec == state)
    # 因果性：只用前 k 个窗口得到的前 k 个判决，必须与全量运行完全一致
    dec_prefix, _ = causal_hmm(z[:150], mean_g, sigma_g, dt=1.0)
    assert np.array_equal(dec_prefix, dec[:150]), "前向滤波被未来污染！"
    # 后 250 个窗口缺失时，前 150 个判决不变（无 backward）
    dec_suffix, _ = causal_hmm(z[150:], mean_g, sigma_g, dt=1.0)
    assert np.array_equal(dec_suffix, dec[150:])
    # 归一化常数来自训练 corr（§5.6）：在听流的 z 恒比另一流高 mean_g
    att = np.ones(500, dtype=int)
    att[250:] = 0
    d_att = mean_g + 0.02 * rng.standard_normal(500)  # z_att − z_other
    z0 = np.where(att == 0, d_att, -d_att) / 2
    z1 = np.where(att == 1, d_att, -d_att) / 2
    corr_train = np.stack([z0, z1], axis=1)
    m, s = normalisation_constants(corr_train, att)
    # 常数应恢复训练数据本身的分布参数（而非默认值）
    assert abs(m - d_att.mean()) < 0.01 and abs(s - d_att.std()) < 0.01, (m, s)
    assert abs(m - mean_g) < 0.05  # 且接近理论 meanDiff


@check("test_submit_roundtrip")
def test_submit():
    """128 Hz 零阶保持提交：写读往返 + 格式约束（§10）。"""
    fs = 32.0
    labels = [np.array([0, 0, 1, 1]), np.array([1, 0])]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sub.csv")
        submit.write_submission(p, labels, source_fs=fs)
        back = submit.read_submission(p)
        assert len(back) == 2
        for a, b in zip(back, labels):
            b2 = submit.zoh_upsample(b, fs)
            assert np.array_equal(a, b2)
        # 128 Hz 输出长度 = 原长 × 4
        assert len(submit.zoh_upsample(labels[0], fs)) == 16
        with open(p) as f:
            header = f.readline().strip()
        assert header == "time_index,label,trial", header


@check("test_expand_lookahead")
def test_expand():
    """expand_to_samples 的 lookahead 语义（§5.7）。"""
    starts = [0, 32]
    decisions = [0, 1]
    n = 128
    seg = hop = 32
    # la = window：判决标注自己的窗
    lab = pipeline.expand_to_samples(starts, decisions, n, seg, hop, 32)
    assert np.array_equal(lab[:32], np.zeros(32, dtype=int))
    assert np.array_equal(lab[32:64], np.ones(32, dtype=int))
    # la = 0：判决只标注窗之后的样本
    lab0 = pipeline.expand_to_samples(starts, decisions, n, seg, hop, 0)
    assert np.array_equal(lab0[:32], np.zeros(32, dtype=int))  # 首窗前填充
    assert np.array_equal(lab0[32:64], np.zeros(32, dtype=int))
    assert np.array_equal(lab0[64:96], np.ones(32, dtype=int))
    # b_k 公式（§5.7）
    la = 10
    labn = pipeline.expand_to_samples([5], [1], 100, seg, hop, la)
    b0 = min(max(5 + seg - la, 0), 100)
    assert labn[b0] == 1 and labn[b0 - 1] == 1 or b0 == 0


@check("test_pipeline_end2end")
def test_pipeline():
    """合成数据端到端：训练→模型存取→评测→HMM 提升（或至少不崩）。"""
    subjects = synthetic.make_dataset(n_subjects=2, n_trials=3, fs=32.0,
                                      n_samples=9600, n_channels=8, seed=7)
    cfg = pipeline.Config(lags=(0.0, 0.1), window=1.0, hop=1.0,
                          lookahead=1.0, target_fs=20.0)
    model = pipeline.train_on_subjects(subjects[:1], cfg)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.npz")
        model.save(p)
        model2 = pipeline.SRModel.load(p)
        assert np.allclose(model.b, model2.b)
    results = pipeline.run_subject(model, subjects[1], cfg)
    trues = np.concatenate([r["true"] for r in results])
    preds = np.concatenate([r["pred"] for r in results])
    acc = metrics.accuracy(trues, preds)
    assert acc > 0.7, "端到端 Acc 过低: %.3f" % acc
    results_hmm = pipeline.run_subject(model, subjects[1], cfg, hmm=True)
    acc_hmm = metrics.accuracy(
        np.concatenate([r["true"] for r in results_hmm]),
        np.concatenate([r["pred"] for r in results_hmm]))
    assert acc_hmm >= acc - 0.02, "HMM 劣化: %.3f < %.3f" % (acc_hmm, acc)
    # 合成切换：Acc 不变（§5.8）
    sw = synthetic.inject_synthetic_switches(subjects[1], 2, seed=3)
    assert sw.synthetic
    r1 = pipeline.run_subject(model, subjects[1], cfg, hmm=False)
    r2 = pipeline.run_subject(model, sw, cfg, hmm=False)
    a1 = metrics.accuracy(np.concatenate([r["true"] for r in r1]),
                          np.concatenate([r["pred"] for r in r1]))
    a2 = metrics.accuracy(np.concatenate([r["true"] for r in r2]),
                          np.concatenate([r["pred"] for r in r2]))
    assert abs(a1 - a2) < 0.02, "合成切换不应改变 Acc: %.3f vs %.3f" % (a1, a2)

    # 流式训练（5 库合并用）必须与一次性训练等价
    by_key = {"A": subjects[:1], "B": subjects[1:2]}
    specs = [("fake", k, "pre_", [1]) for k in by_key]

    def loader(_root, key, _j, _prefix):
        import copy as _copy
        return _copy.deepcopy(by_key[key][0])

    model_s = pipeline.train_streaming(specs, cfg, loader=loader, log=lambda *a: None)
    assert model_s.channels == model.channels, (model_s.channels, model.channels)
    r = np.corrcoef(model_s.b, model.b)[0, 1]
    assert r > 0.99, "流式训练权重与一次性训练不一致 corr=%.4f" % r
    assert abs(model_s.mean_global - model.mean_global) < 0.05
    res_s = pipeline.run_subject(model_s, subjects[1], cfg)
    acc_s = metrics.accuracy(np.concatenate([x["true"] for x in res_s]),
                             np.concatenate([x["pred"] for x in res_s]))
    assert acc_s > 0.7, "流式模型 Acc 过低: %.3f" % acc_s

    # 前缀自动探测：DTU 是 pre_（32 Hz），其余数据集无前缀（128 Hz）
    with tempfile.TemporaryDirectory() as td:
        for name in ("pre_dataSub1.mat", "pre_dataStim1.mat"):
            open(os.path.join(td, name), "w").close()
        _e, _s, used = cnd._resolve_pair(td, 1, "")
        assert used == "pre_", used
        for name in ("dataSub2.mat", "dataStim2.mat"):
            open(os.path.join(td, name), "w").close()
        _e, _s, used = cnd._resolve_pair(td, 2, "pre_")
        assert used == "", used


def main():
    passed = 0
    for name, fn in RESULTS:
        try:
            fn()
            print("PASS  %s" % name)
            passed += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL  %s: %r" % (name, e))
    print("%d/%d passed" % (passed, len(RESULTS)))
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
