"""Every number quoted in the README, recomputed from the shipped artifacts.

Standalone: needs only numpy, pandas, scipy, scikit-learn and matplotlib, no GPU.

    python scripts/step12_paper_stats.py                      # uses ./step10_artifacts
    python scripts/step12_paper_stats.py --artifacts DIR --figures DIR --margin 0.0068

Reads runs/*.json (manifests), preds/*_test.npz (test logits + labels) and
curves_v2/*.json (per-epoch dev curves). Prints each table and writes the README
figures. Only TEST logits were saved during training, so temperature scaling here is
cross-fitted on the test set (fit T on one half, score ECE on the other, swap and
average). It is a sensitivity check, not a dev-fitted calibration result.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

METHODS = ("full_ft", "lora")
LABEL = {"full_ft": "Full fine-tuning", "lora": "LoRA"}
N_BINS = 15
BOOT_N = 10000


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, n_bins=N_BINS):
    """Expected calibration error, equal-width confidence bins on the top-1 prob."""
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def fit_temperature(logits, y):
    def nll(t):
        p = softmax(logits / t)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x


def crossfit_ece(logits, y, seed=0):
    """Two-fold cross-fitted temperature scaling on the test set."""
    idx = np.random.default_rng(seed).permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    out = []
    for fit, ev in ((a, b), (b, a)):
        t = fit_temperature(logits[fit], y[fit])
        out.append(ece(softmax(logits[ev] / t), y[ev]))
    return float(np.mean(out))


def paired_t_ci(d, level=0.95):
    d = np.asarray(d)
    half = stats.t.ppf(0.5 + level / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean() - half, d.mean() + half


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_exact(right_a, right_b):
    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    return b, c, stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def load(art, run_id):
    f = np.load(f"{art}/preds/{run_id}_test.npz")
    return f["logits"].astype(np.float64), f["labels"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifacts", default="step10_artifacts")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--margin", type=float, default=0.0068,
                    help="TOST equivalence margin on macro-F1 (default: the "
                         "reproducibility floor quoted in the README)")
    args = ap.parse_args()
    art = args.artifacts

    runs = pd.DataFrame([json.load(open(p)) for p in sorted(glob.glob(f"{art}/runs/*.json"))])
    if runs.empty:
        raise SystemExit(f"no manifests under {art}/runs")
    fractions = sorted(runs.fraction.unique())
    seeds = sorted(runs.seed.unique())
    print(f"{len(runs)} runs, fractions {fractions}, seeds {seeds}")
    print("\nruns per GPU and library version:")
    print(runs.groupby(["gpu", "transformers", "peft", "precision"]).size().to_string())

    # ---- per-run test metrics from the logits --------------------------------
    rows, preds = [], {}
    for _, r in runs.iterrows():
        logits, y = load(art, r.run_id)
        p = softmax(logits)
        preds[r.run_id] = (logits, p, y)
        rows.append(dict(run_id=r.run_id, method=r.method, fraction=r.fraction, seed=r.seed,
                         gpu=r.gpu, best_epoch=r.best_epoch, ceiling_hit=r.ceiling_hit,
                         f1=f1_score(y, p.argmax(1), average="macro"),
                         acc=(p.argmax(1) == y).mean(), conf=p.max(1).mean(),
                         ece=ece(p, y), ece_ts=crossfit_ece(logits, y)))
    m = pd.DataFrame(rows)
    wide = m.pivot_table(index=["fraction", "seed"], columns="method",
                         values=["f1", "ece", "ece_ts", "best_epoch"])

    split = runs.groupby(["fraction", "seed"]).gpu.nunique()
    print(f"(fraction, seed) pairs split across GPUs: {int((split > 1).sum())}")

    # ---- accuracy + TOST -----------------------------------------------------
    print(f"\n=== test macro-F1 (TOST margin +/-{args.margin}) ===")
    for f in fractions:
        a, b = wide.loc[f, ("f1", "full_ft")], wide.loc[f, ("f1", "lora")]
        d = b - a
        lo, hi = paired_t_ci(d)
        lo90, hi90 = paired_t_ci(d, 0.90)
        equiv = lo90 > -args.margin and hi90 < args.margin
        print(f"{f:<5g} full {a.mean():.4f} (sd {a.std():.4f})  lora {b.mean():.4f} "
              f"(sd {b.std():.4f})  gap {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"p={stats.ttest_rel(b, a).pvalue:.3f}  90% CI [{lo90:+.4f}, {hi90:+.4f}]  "
              f"TOST {'EQUIVALENT' if equiv else 'not shown'}  "
              f"(smallest margin that passes: {max(-lo90, hi90):.4f})")

    # ---- per-seed McNemar with Holm -----------------------------------------
    print("\n=== per-seed McNemar on accuracy, Holm-corrected over all pairs ===")
    tests = []
    for f in fractions:
        for s in seeds:
            _, pf, y = preds[f"full_ft_frac{f:g}_seed{s}"]
            _, pl, _ = preds[f"lora_frac{f:g}_seed{s}"]
            b, c, p = mcnemar_exact(pf.argmax(1) == y, pl.argmax(1) == y)
            tests.append((f, s, b, c, p, (pf.argmax(1) == pl.argmax(1)).mean()))
    adj = holm(np.array([t[4] for t in tests]))
    for (f, s, b, c, p, _), pa in zip(tests, adj):
        flag = "  <- p<0.05" if p < 0.05 else ""
        print(f"{f:<5g} seed {s:<3} full-only right {b:4d}  lora-only right {c:4d}  "
              f"p={p:.4f}  holm={pa:.4f}{flag}")
    print(f"{sum(t[4] < 0.05 for t in tests)} of {len(tests)} raw p<0.05; "
          f"{int((adj < 0.05).sum())} survive Holm")

    print("\n=== prediction agreement (same seed) ===")
    agree = pd.DataFrame(tests, columns=["fraction", "seed", "b", "c", "p", "agree"])
    print(agree.groupby("fraction")[["agree", "b", "c"]]
          .agg({"agree": "mean", "b": "sum", "c": "sum"}).round(4).to_string())

    # ---- calibration ---------------------------------------------------------
    rng = np.random.default_rng(0)
    print(f"\n=== calibration: ECE ({N_BINS} bins), paired over seeds ===")
    for f in fractions:
        ef, el = wide.loc[f, ("ece", "full_ft")], wide.loc[f, ("ece", "lora")]
        d = (el - ef).values
        boot = rng.choice(d, (BOOT_N, len(d))).mean(1)
        tf, tl = wide.loc[f, ("ece_ts", "full_ft")], wide.loc[f, ("ece_ts", "lora")]
        dt = (tl - tf).values
        lo_t, hi_t = paired_t_ci(dt)
        sub = m[m.fraction == f].groupby("method")[["conf", "acc"]].mean()
        print(f"{f:<5g} full {ef.mean():.3f} (sd {ef.std():.3f})  lora {el.mean():.3f} "
              f"(sd {el.std():.3f})  dECE {d.mean():+.3f}  p={stats.ttest_rel(el, ef).pvalue:.2g}  "
              f"t-CI [{paired_t_ci(d)[0]:+.3f}, {paired_t_ci(d)[1]:+.3f}]  "
              f"seed-bootstrap CI [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]  "
              f"dz={d.mean() / d.std(ddof=1):.1f}  seeds LoRA better {int((d < 0).sum())}/{len(d)}")
        print(f"      mean confidence full {sub.loc['full_ft', 'conf']:.3f} / lora "
              f"{sub.loc['lora', 'conf']:.3f} at accuracy {sub.loc['full_ft', 'acc']:.3f} / "
              f"{sub.loc['lora', 'acc']:.3f}")
        print(f"      after cross-fitted temperature scaling: full {tf.mean():.3f}  lora "
              f"{tl.mean():.3f}  dECE {dt.mean():+.3f}  t-CI [{lo_t:+.3f}, {hi_t:+.3f}]")

    # ---- convergence ---------------------------------------------------------
    print("\n=== best (dev-selected) epoch ===")
    for f in fractions:
        be = wide.loc[f, "best_epoch"]
        d = be["lora"] - be["full_ft"]
        p = stats.ttest_rel(be["lora"], be["full_ft"]).pvalue if d.std() > 0 else float("nan")
        hits = m[m.fraction == f].groupby("method").ceiling_hit.sum()
        print(f"{f:<5g} full {sorted(be['full_ft'].astype(int))} (mean {be['full_ft'].mean():.1f})  "
              f"lora {sorted(be['lora'].astype(int))} (mean {be['lora'].mean():.1f})  "
              f"paired gap {d.mean():+.1f}  LoRA later in {int((d > 0).sum())}/{len(d)}  "
              f"p={p:.4f}  ceiling hit full {int(hits['full_ft'])}/5, lora {int(hits['lora'])}/5")

    # ---- efficiency (never pooled across GPUs) -------------------------------
    print("\n=== efficiency by GPU ===")
    ok = runs[runs.get("wall_clock_comparable", True).astype(bool)]
    print(ok.groupby(["gpu", "method"]).agg(
        runs=("run_id", "count"), trainable=("trainable_params", "max"),
        model_mb=("saved_model_mb", "mean"), peak_mb=("peak_gpu_mem_mb", "mean"),
        samples_per_s=("train_samples_per_s", "mean")).round(1).to_string())

    figures(art, args.figures, m, fractions)


def figures(art, out, m, fractions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    colors = {"full_ft": "#1f77b4", "lora": "#ff7f0e"}
    x = np.arange(len(fractions))
    xt = [f"{f:.0%}" for f in fractions]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, ylabel, title in (
            (axes[0], "f1", "test macro-F1", "Accuracy: no measurable gap"),
            (axes[1], "ece", f"test ECE ({N_BINS} bins, lower is better)",
             "Calibration: gap largest at 1%")):
        for j, meth in enumerate(METHODS):
            g = m[m.method == meth].groupby("fraction")[col]
            off = (j - 0.5) * 0.18
            ax.errorbar(x + off, g.mean().values, yerr=g.std().values, fmt="o",
                        color=colors[meth], capsize=4, label=LABEL[meth])
            for i, f in enumerate(fractions):
                v = m[(m.method == meth) & (m.fraction == f)][col]
                ax.scatter(np.full(len(v), i + off + 0.05), v, s=8,
                           color=colors[meth], alpha=0.35)
        ax.set_xticks(x, xt)
        ax.set_xlabel("fraction of XNLI-bn training data")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    fig.suptitle("5 seeds per point; bars = sd, dots = individual seeds", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out}/f1_and_ece.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fractions), figsize=(4 * len(fractions), 3.4),
                             sharey=False)
    for ax, f in zip(np.atleast_1d(axes), fractions):
        for meth in METHODS:
            for p in sorted(glob.glob(f"{art}/curves_v2/{meth}_frac{f:g}_seed*.json")):
                c = json.load(open(p))
                ax.plot([e["epoch"] for e in c], [e["eval_f1_macro"] for e in c],
                        color=colors[meth], alpha=0.6, lw=1.2)
            ax.plot([], [], color=colors[meth], label=LABEL[meth])
        ax.set_title(f"{f:.0%} of training data", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("validation macro-F1")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{out}/dev_curves.png", dpi=160)
    plt.close(fig)
    print(f"\nwrote {out}/f1_and_ece.png and {out}/dev_curves.png")


if __name__ == "__main__":
    main()
