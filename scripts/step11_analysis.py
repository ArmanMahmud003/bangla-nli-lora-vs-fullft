# Standalone copy of notebook cell 9 (notebooks/Step10_retrain_ColabA100.ipynb).
# Run from the command line against the shipped artifacts:
#     PROJECT_DIR=step10_artifacts python scripts/step11_analysis.py
# The README's headline statistics come from scripts/step12_paper_stats.py.
import os as _os
if "PROJECT_DIR" not in globals():
    PROJECT_DIR = _os.environ.get("PROJECT_DIR", "step10_artifacts")

# =============================================================================
# STEP 11 - analysis: test-set tables, significance, efficiency, figures
# =============================================================================
# Reads Step 10's outputs (runs/*.json + preds/*.npz). No GPU, no training.
# Same logic as the v1 analysis cell, so numbers stay comparable across notebooks.
# The README's full statistics (TOST, Holm, ECE, temperature scaling) come from
# scripts/step12_paper_stats.py, which runs on the exported artifacts.
# =============================================================================

import glob, itertools, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import binomtest

RUNS_DIR   = f"{PROJECT_DIR}/runs"
PREDS_DIR  = f"{PROJECT_DIR}/preds"
FIG_DIR    = f"{PROJECT_DIR}/figures"
OLD_LOG    = f"{PROJECT_DIR}/results_log.csv"
BOOT_N     = 2000
os.makedirs(FIG_DIR, exist_ok=True)

runs = [json.load(open(p)) for p in sorted(glob.glob(f"{RUNS_DIR}/*.json"))]
if not runs:
    raise SystemExit("no runs/*.json yet - run Step 10 first")
df = pd.DataFrame(runs).sort_values(["fraction", "method", "seed"]).reset_index(drop=True)
print(f"{len(df)} runs: "
      + ", ".join(f"{f:g}={n}" for f, n in df.groupby('fraction').size().items()))

# ---- 1. consistency checks -------------------------------------------------
print("\n=== consistency ===")
problems = []
for col in ["precision", "batch_size", "epoch_ceiling", "patience_epochs"]:
    if col in df.columns and df[col].nunique(dropna=False) > 1:
        problems.append(f"{col} differs across runs: "
                        f"{sorted(df[col].unique(), key=str)}")
for frac, g in df.groupby("fraction"):
    for col in ["patience_evals", "evals_per_epoch"]:
        if col in g.columns and g[col].nunique(dropna=False) > 1:
            problems.append(f"fraction {frac:g}: {col} differs within fraction")
for (frac, method), g in df.groupby(["fraction", "method"]):
    if g["seed"].duplicated().any():
        problems.append(f"duplicate seeds at fraction {frac}, {method}")
for frac, g in df.groupby("fraction"):
    per = g.groupby("method")["seed"].apply(set)
    if len(per) < 2:
        problems.append(f"fraction {frac}: only {list(per.index)} - no comparison")
    elif len(per) == 2 and per.iloc[0] != per.iloc[1]:
        problems.append(f"fraction {frac}: seeds differ between methods")
missing_preds = [r for r in df["run_id"]
                 if not os.path.exists(f"{PREDS_DIR}/{r}_test.npz")]
if missing_preds:
    problems.append(f"no saved predictions for {missing_preds}")
hit = df[df["ceiling_hit"]]
if len(hit):
    problems.append(f"ceiling still binding for {list(hit['run_id'])}")
drift = df[df["dev_reload_drift"] > 1e-3]
if len(drift):
    problems.append(f"best model did not reload cleanly for {list(drift['run_id'])}")
# [v2] reload verification check
if "reload_verified" in df.columns:
    unver = df[~df["reload_verified"].fillna(True).astype(bool)]
    if len(unver):
        problems.append(f"reload verification failed for {list(unver['run_id'])}")
    big_drift = df[(df.get("reload_max_logit_drift").fillna(0) > 1e-3)]
    if len(big_drift):
        problems.append(f"logit drift >1e-3 on reload for {list(big_drift['run_id'])}")

MACHINE = [c for c in ("gpu", "torch", "transformers", "peft") if c in df.columns]
multi_machine = {c: sorted(df[c].dropna().unique()) for c in MACHINE
                 if df[c].nunique(dropna=False) > 1}
for (frac, seed), g in df.groupby(["fraction", "seed"]):
    for c in multi_machine:
        if g[c].nunique(dropna=False) > 1:
            problems.append(f"fraction {frac:g} seed {seed} split across "
                            f"{sorted(g[c].dropna().unique(), key=str)} ({c})")
print("\n".join(f"  [!] {p}" for p in problems) if problems else "  nothing to flag")
if multi_machine:
    for c, vals in multi_machine.items():
        print(f"  note: {c} varies: {vals}")

rule = df["patience_epochs"].dropna().unique()
print("  stopping rule: " + (f"early stop patience {rule[0]:g}" if len(rule)
      else f"none - {df['epoch_ceiling'].max():g} full epochs, best on val"))
print(f"  early stopped: {int(df['early_stopped'].sum())}/{len(df)} "
      f"({dict(df.groupby('method')['early_stopped'].sum())})")

# ---- 2. test scores by fraction and method --------------------------------
print("\n=== TEST macro-F1 (mean +/- sd over seeds) ===")
agg = (df.groupby(["fraction", "method"])[["test_f1_macro", "test_accuracy"]]
         .agg(["mean", "std", "count"]).round(4))
print(agg.to_string())

# ---- 3. paired per-seed gaps ----------------------------------------------
print("\n=== paired gaps, LoRA minus full FT (per seed) ===")
wide = df.pivot_table(index=["fraction", "seed"], columns="method",
                      values="test_f1_macro")
pairs = wide.dropna(subset=[c for c in ("lora", "full_ft") if c in wide.columns])
if {"lora", "full_ft"}.issubset(pairs.columns):
    pairs = pairs.assign(gap=(pairs["lora"] - pairs["full_ft"]).round(4))
    print(pairs.round(4).to_string())
    summary = pairs.groupby("fraction")["gap"].agg(
        ["mean", "std", "count", lambda s: int((s > 0).sum())])
    summary.columns = ["mean_gap", "sd", "n_seeds", "seeds_lora_ahead"]
    print("\n" + summary.round(4).to_string())
else:
    print("  need both methods at the same fraction+seed to pair")

# ---- 4. significance ------------------------------------------------------
def load_preds(run_id):
    z = np.load(f"{PREDS_DIR}/{run_id}_test.npz")
    return z["logits"].astype(np.float32).argmax(1), z["labels"]

def mcnemar(a_pred, b_pred, y):
    a, b_ = (a_pred == y), (b_pred == y)
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p

def boot_f1_gap(a_pred, b_pred, y, n=BOOT_N, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.array([f1_score(y[i], a_pred[i], average="macro")
                  - f1_score(y[i], b_pred[i], average="macro") for i in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

print("\n=== significance, per fraction and seed (LoRA vs full FT) ===")
print("  b = LoRA right / full FT wrong; c = reverse; p = exact McNemar.")
sig_rows = []
for (frac, seed), g in df.groupby(["fraction", "seed"]):
    have = dict(zip(g["method"], g["run_id"]))
    if not {"lora", "full_ft"}.issubset(have): continue
    if any(not os.path.exists(f"{PREDS_DIR}/{have[m]}_test.npz")
           for m in ("lora", "full_ft")): continue
    lp, y = load_preds(have["lora"])
    fp, y2 = load_preds(have["full_ft"])
    assert np.array_equal(y, y2), "test labels differ - wrong preds file"
    b, c, p = mcnemar(lp, fp, y)
    f1_l = f1_score(y, lp, average="macro")
    f1_f = f1_score(y, fp, average="macro")
    lo, hi = boot_f1_gap(lp, fp, y)
    sig_rows.append({"fraction": frac, "seed": seed,
                     "f1_lora": round(f1_l, 4), "f1_full_ft": round(f1_f, 4),
                     "f1_gap": round(f1_l - f1_f, 4),
                     "gap_ci95": f"[{lo:+.4f}, {hi:+.4f}]",
                     "ci_excludes_0": (lo > 0) or (hi < 0),
                     "mcnemar_b": b, "mcnemar_c": c,
                     "mcnemar_p": round(p, 4), "sig_05": p < 0.05})
sig = pd.DataFrame(sig_rows)
if len(sig):
    print(sig.to_string(index=False))
    sig.to_csv(f"{PROJECT_DIR}/significance.csv", index=False)
    n_sig = int(sig["sig_05"].sum())
    print(f"\n  {n_sig} of {len(sig)} comparisons reach p<0.05 on accuracy; "
          f"{int(sig['ci_excludes_0'].sum())} have F1 CI excluding zero.")
else:
    print("  no paired prediction files yet")

# ---- 5. efficiency --------------------------------------------------------
print("\n=== efficiency (what LoRA is actually for) ===")
if "wall_clock_comparable" in df.columns:
    df["wall_clock_comparable"] = df["wall_clock_comparable"].fillna(True).astype(bool)
timed = df[df["wall_clock_comparable"]] if "wall_clock_comparable" in df.columns else df
if "wall_clock_comparable" in df.columns and len(timed) < len(df):
    print(f"  {len(df) - len(timed)} resumed run(s) excluded from timing")
by = ["method", "gpu"] if ("gpu" in df.columns and df["gpu"].nunique() > 1) \
     else ["method"]
eff = (df.groupby(by)
         .agg(runs=("run_id", "count"),
              trainable_params=("trainable_params", "max"),
              pct_trainable=("pct_trainable", "max"),
              saved_model_mb=("saved_model_mb", "mean"),
              peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean")).round(2))
eff = eff.join(timed.groupby(by).agg(
    timed_runs=("run_id", "count"),
    samples_per_s=("train_samples_per_s", "mean")).round(2))
print(eff.to_string())
print("\nwall-clock minutes per run, by fraction:")
idx = ["gpu", "fraction"] if len(by) == 2 else "fraction"
print((timed.pivot_table(index=idx, columns="method",
                         values="train_runtime_s", aggfunc="mean") / 60)
      .round(1).to_string())

# ---- 6. figures -----------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
for method, style in (("full_ft", "o-"), ("lora", "s--")):
    g = df[df["method"] == method].groupby("fraction")["test_f1_macro"]
    if not len(g): continue
    ax.errorbar(g.mean().index, g.mean().values,
                yerr=g.std().fillna(0).values, fmt=style, capsize=3,
                label="full fine-tuning" if method == "full_ft" else "LoRA")
ax.set_xscale("log")
ax.set_xlabel("fraction of training data (log scale)")
ax.set_ylabel("test macro-F1")
ax.set_title("Matched epoch budget, selection on validation, reported on test")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG_DIR}/test_f1_vs_fraction.{ext}", dpi=200)
print(f"\nwrote {FIG_DIR}/test_f1_vs_fraction.png/.pdf")

fig2, ax2 = plt.subplots(figsize=(6, 4))
for _, r in df.iterrows():
    cp = f"{PROJECT_DIR}/curves_v2/{r['run_id']}.json"
    if not os.path.exists(cp): continue
    c = json.load(open(cp))
    ax2.plot([e["epoch"] for e in c], [e["eval_f1_macro"] for e in c],
             ("s--" if r["method"] == "lora" else "o-"), alpha=0.7,
             color=("tab:orange" if r["method"] == "lora" else "tab:blue"),
             label=r["method"] if r["run_id"].endswith(str(df["seed"].iloc[0])) else None)
ax2.set_xlabel("epoch"); ax2.set_ylabel("validation macro-F1")
ax2.set_title("Convergence at a matched ceiling")
ax2.grid(alpha=0.3)
handles, labels = ax2.get_legend_handles_labels()
if labels:
    ax2.legend(dict(zip(labels, handles)).values(),
               dict(zip(labels, handles)).keys())
fig2.tight_layout()
for ext in ("png", "pdf"):
    fig2.savefig(f"{FIG_DIR}/dev_curves_matched.{ext}", dpi=200)
print(f"wrote {FIG_DIR}/dev_curves_matched.png/.pdf")

if os.path.exists(OLD_LOG):
    old = pd.read_csv(OLD_LOG)
    keep = old[~old["fraction"].isin(df["fraction"].unique())]
    if len(keep):
        print("\n=== first pass, DEV only (label as dev, unmatched budgets) ===")
        print(keep.pivot_table(index="fraction", columns="method",
                               values="f1_macro", aggfunc="mean").round(4).to_string())
