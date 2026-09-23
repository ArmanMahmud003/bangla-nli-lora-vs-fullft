# =============================================================================
# STEP 11 - analysis: test-set tables, significance tests, efficiency, figures
# =============================================================================
# Reads what Step 10 wrote (runs/*.json + preds/*.npz). No GPU, no training.
# Produces, in order:
#   1. consistency checks - anything that would make the comparison unfair
#   2. TEST macro-F1 by fraction and method, mean +/- sd over seeds
#   3. paired per-seed gaps (the honest way to compare when seeds are shared)
#   4. McNemar exact test on accuracy + paired bootstrap CI on macro-F1
#   5. efficiency table: trainable params, wall-clock, peak memory, model size
#   6. figures into figures/
# Needs from earlier cells: PROJECT_DIR (and Step 10's constants if run in the
# same session; otherwise the paths below are self-contained)
# =============================================================================

import glob, itertools, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import binomtest

RUNS_DIR   = f"{PROJECT_DIR}/runs"
PREDS_DIR  = f"{PROJECT_DIR}/preds"
FIG_DIR    = f"{PROJECT_DIR}/figures"
OLD_LOG    = f"{PROJECT_DIR}/results_log.csv"      # the first pass, dev-only
BOOT_N     = 2000                                   # bootstrap resamples
os.makedirs(FIG_DIR, exist_ok=True)

runs = [json.load(open(p)) for p in sorted(glob.glob(f"{RUNS_DIR}/*.json"))]
if not runs:
    raise SystemExit("no runs/*.json yet - run Step 10 first")
df = pd.DataFrame(runs).sort_values(["fraction", "method", "seed"]).reset_index(drop=True)
print(f"{len(df)} runs: "
      + ", ".join(f"{f:g}={n}" for f, n in df.groupby('fraction').size().items()))

# PLACEHOLDER_A

# ---- 1. consistency checks --------------------------------------------------
print("\n=== consistency ===")
problems = []
# these must be identical everywhere, or the two methods were not run under the
# same rules. patience_evals is NOT one of them: it is patience_epochs x evals per
# epoch, so it legitimately differs between fractions - check it per fraction.
# dropna=False matters: patience_epochs is null when early stopping is off, and a
# grid half-trained with early stopping and half without must be caught.
for col in ["precision", "batch_size", "epoch_ceiling", "patience_epochs"]:
    if col in df.columns and df[col].nunique(dropna=False) > 1:
        problems.append(f"{col} differs across runs: "
                        f"{sorted(df[col].unique(), key=str)} - "
                        f"runs are not comparable unless this is deliberate")
for frac, g in df.groupby("fraction"):
    for col in ["patience_evals", "evals_per_epoch"]:
        if col in g.columns and g[col].nunique(dropna=False) > 1:
            problems.append(f"fraction {frac:g}: {col} differs within the fraction "
                            f"({sorted(g[col].unique(), key=str)}) - selection grids differ")
for (frac, method), g in df.groupby(["fraction", "method"]):
    if g["seed"].duplicated().any():
        problems.append(f"duplicate seeds at fraction {frac}, {method}")
for frac, g in df.groupby("fraction"):
    per = g.groupby("method")["seed"].apply(set)
    if len(per) < 2:
        problems.append(f"fraction {frac}: only {list(per.index)} present - no comparison")
    elif len(per) == 2 and per.iloc[0] != per.iloc[1]:
        problems.append(f"fraction {frac}: seeds differ between methods "
                        f"({per.iloc[0]} vs {per.iloc[1]}) - gaps are not paired")
missing_preds = [r for r in df["run_id"] if not os.path.exists(f"{PREDS_DIR}/{r}_test.npz")]
if missing_preds:
    problems.append(f"no saved predictions for {missing_preds} - no significance test")
hit = df[df["ceiling_hit"]]
if len(hit):
    problems.append(f"ceiling still binding for {list(hit['run_id'])} - the budget "
                    f"limited these runs, so say so or raise the ceiling")
drift = df[df["dev_reload_drift"] > 1e-3]
if len(drift):
    problems.append(f"best model did not reload cleanly for {list(drift['run_id'])}")
# Runs may legitimately come from several machines - accuracy is a property of the
# training recipe, not of the card. But a PAIR must not be split: the per-seed gap is
# LoRA minus full FT at the same fraction and seed, so if one side ran on a different
# GPU (or torch build) that gap mixes hardware with method.
MACHINE = [c for c in ("gpu", "torch", "transformers", "peft") if c in df.columns]
multi_machine = {c: sorted(df[c].dropna().unique()) for c in MACHINE
                 if df[c].nunique(dropna=False) > 1}
for (frac, seed), g in df.groupby(["fraction", "seed"]):
    for c in multi_machine:
        if g[c].nunique(dropna=False) > 1:
            problems.append(f"fraction {frac:g} seed {seed} was split across "
                            f"{sorted(g[c].dropna().unique(), key=str)} ({c}) - run both "
                            f"methods of a pair on one machine, or the gap is confounded")
print("\n".join(f"  [!] {p}" for p in problems) if problems else "  nothing to flag")
if multi_machine:
    for c, vals in multi_machine.items():
        print(f"  note: {c} varies across runs: {vals}. Fine for accuracy as long as no "
              f"pair is split; wall-clock and memory are only comparable within one.")
rule = df["patience_epochs"].dropna().unique()
print("  stopping rule: " + (f"early stopping, patience {rule[0]:g} epoch(s)" if len(rule)
      else f"none - every run trained the full {df['epoch_ceiling'].max():g} epochs, "
           "best epoch chosen on validation"))
print(f"  early stopped: {int(df['early_stopped'].sum())}/{len(df)} "
      f"({dict(df.groupby('method')['early_stopped'].sum())})")

# ---- 2. test scores by fraction and method ----------------------------------
print("\n=== TEST macro-F1 (mean +/- sd over seeds) ===")
agg = (df.groupby(["fraction", "method"])[["test_f1_macro", "test_accuracy"]]
         .agg(["mean", "std", "count"]).round(4))
print(agg.to_string())

# PLACEHOLDER_B

# ---- 3. paired per-seed gaps -------------------------------------------------
# Both methods share the seed's data subset, so the per-seed difference is a
# paired measurement: report those, not just two independent means.
print("\n=== paired gaps, LoRA minus full FT (per seed) ===")
wide = df.pivot_table(index=["fraction", "seed"], columns="method",
                      values="test_f1_macro")
pairs = wide.dropna(subset=[c for c in ("lora", "full_ft") if c in wide.columns])
if {"lora", "full_ft"}.issubset(pairs.columns):
    pairs = pairs.assign(gap=(pairs["lora"] - pairs["full_ft"]).round(4))
    print(pairs.round(4).to_string())
    summary = pairs.groupby("fraction")["gap"].agg(["mean", "std", "count",
                                                    lambda s: int((s > 0).sum())])
    summary.columns = ["mean_gap", "sd", "n_seeds", "seeds_lora_ahead"]
    print("\n" + summary.round(4).to_string())
else:
    print("  need both methods at the same fraction+seed to pair")

# ---- 4. significance --------------------------------------------------------
def load_preds(run_id):
    z = np.load(f"{PREDS_DIR}/{run_id}_test.npz")
    return z["logits"].astype(np.float32).argmax(1), z["labels"]

def mcnemar(a_pred, b_pred, y):
    """exact test on per-example correctness: does either method win more often?"""
    a, b_ = (a_pred == y), (b_pred == y)
    b = int((a & ~b_).sum())          # first method right, second wrong
    c = int((~a & b_).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    return b, c, p

def boot_f1_gap(a_pred, b_pred, y, n=BOOT_N, seed=0):
    """paired bootstrap over test examples: 95% CI for the macro-F1 difference.
    McNemar tests accuracy; your headline metric is macro-F1, so test that too."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.array([f1_score(y[i], a_pred[i], average="macro")
                  - f1_score(y[i], b_pred[i], average="macro") for i in idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

# PLACEHOLDER_C

print("\n=== significance, per fraction and seed (LoRA vs full FT) ===")
print("  b = LoRA right / full FT wrong, c = the reverse; p = exact McNemar.")
sig_rows = []
for (frac, seed), g in df.groupby(["fraction", "seed"]):
    have = dict(zip(g["method"], g["run_id"]))
    if not {"lora", "full_ft"}.issubset(have):
        continue
    if any(not os.path.exists(f"{PREDS_DIR}/{have[m]}_test.npz") for m in ("lora", "full_ft")):
        continue
    lp, y = load_preds(have["lora"])
    fp, y2 = load_preds(have["full_ft"])
    assert np.array_equal(y, y2), "test labels differ between runs - wrong preds file"
    b, c, p = mcnemar(lp, fp, y)
    f1_l = f1_score(y, lp, average="macro")
    f1_f = f1_score(y, fp, average="macro")
    lo, hi = boot_f1_gap(lp, fp, y)
    sig_rows.append({"fraction": frac, "seed": seed, "f1_lora": round(f1_l, 4),
                     "f1_full_ft": round(f1_f, 4), "f1_gap": round(f1_l - f1_f, 4),
                     "gap_ci95": f"[{lo:+.4f}, {hi:+.4f}]",
                     "ci_excludes_0": (lo > 0) or (hi < 0),
                     "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p, 4),
                     "sig_05": p < 0.05})
sig = pd.DataFrame(sig_rows)
if len(sig):
    print(sig.to_string(index=False))
    sig.to_csv(f"{PROJECT_DIR}/significance.csv", index=False)
    n_sig = int(sig["sig_05"].sum())
    print(f"\n  {n_sig} of {len(sig)} seed-level comparisons reach p<0.05 on accuracy; "
          f"{int(sig['ci_excludes_0'].sum())} have a macro-F1 CI that excludes zero.")
    print("  A gap that fails both is noise: say 'no measurable difference', not "
          "'LoRA was slightly better'.")
else:
    print("  no paired prediction files yet")

# ---- 5. efficiency ----------------------------------------------------------
# Parameter counts and file sizes are properties of the method. Wall-clock and peak
# memory are properties of the method ON THIS GPU, so they are grouped by GPU when
# more than one was used - averaging a T4 and a 5060 together is meaningless.
# A run that was resumed after an interruption only timed the part after the crash,
# so it is dropped from the speed columns here and kept everywhere else.
print("\n=== efficiency (what LoRA is actually for) ===")
if "wall_clock_comparable" in df.columns:      # older manifests predate the column
    df["wall_clock_comparable"] = df["wall_clock_comparable"].fillna(True).astype(bool)
timed = df[df["wall_clock_comparable"]] if "wall_clock_comparable" in df.columns else df
if "wall_clock_comparable" in df.columns and len(timed) < len(df):
    print(f"  {len(df) - len(timed)} resumed run(s) excluded from the timing columns "
          f"(their clock only covers the part after the interruption): "
          f"{', '.join(df.loc[~df['wall_clock_comparable'], 'run_id'])}")
by = ["method", "gpu"] if ("gpu" in df.columns and df["gpu"].nunique() > 1) else ["method"]
eff = (df.groupby(by)
         .agg(runs=("run_id", "count"),
              trainable_params=("trainable_params", "max"),
              pct_trainable=("pct_trainable", "max"),
              saved_model_mb=("saved_model_mb", "mean"),
              peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean")).round(2))
eff = eff.join(timed.groupby(by).agg(timed_runs=("run_id", "count"),
                                     samples_per_s=("train_samples_per_s", "mean")).round(2))
print(eff.to_string())
print("\nwall-clock minutes per run, by fraction:")
idx = ["gpu", "fraction"] if len(by) == 2 else "fraction"
print((timed.pivot_table(index=idx, columns="method", values="train_runtime_s",
                         aggfunc="mean") / 60).round(1).to_string())
if len(by) == 2:
    print("Quote speed and memory from ONE machine and name it; the accuracy tables above "
          "pool every run, which is correct, but these two columns do not pool.")

# PLACEHOLDER_D

# ---- 6. figures -------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
for method, style in (("full_ft", "o-"), ("lora", "s--")):
    g = df[df["method"] == method].groupby("fraction")["test_f1_macro"]
    if not len(g):
        continue
    ax.errorbar(g.mean().index, g.mean().values,
                yerr=g.std().fillna(0).values, fmt=style, capsize=3,
                label="full fine-tuning" if method == "full_ft" else "LoRA")
ax.set_xscale("log")
ax.set_xlabel("fraction of training data (log scale)")
ax.set_ylabel("test macro-F1")
ax.set_title("Matched epoch budget, selection on validation, reported on test")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{FIG_DIR}/test_f1_vs_fraction.{ext}", dpi=200)
print(f"\nwrote {FIG_DIR}/test_f1_vs_fraction.png/.pdf")

# dev curves for the matched runs: shows WHY the old comparison was unfair
fig2, ax2 = plt.subplots(figsize=(6, 4))
for _, r in df.iterrows():
    cp = f"{PROJECT_DIR}/curves_v2/{r['run_id']}.json"
    if not os.path.exists(cp):
        continue
    c = json.load(open(cp))
    ax2.plot([e["epoch"] for e in c], [e["eval_f1_macro"] for e in c],
             ("s--" if r["method"] == "lora" else "o-"), alpha=0.7,
             color=("tab:orange" if r["method"] == "lora" else "tab:blue"),
             label=r["method"] if r["run_id"].endswith(str(df["seed"].iloc[0])) else None)
ax2.set_xlabel("epoch")
ax2.set_ylabel("validation macro-F1")
ax2.set_title("Convergence at a matched ceiling")
ax2.grid(alpha=0.3)
handles, labels = ax2.get_legend_handles_labels()
if labels:
    ax2.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
fig2.tight_layout()
for ext in ("png", "pdf"):
    fig2.savefig(f"{FIG_DIR}/dev_curves_matched.{ext}", dpi=200)
print(f"wrote {FIG_DIR}/dev_curves_matched.png/.pdf")

# ---- the old first pass, for the fractions you are not retraining ------------
if os.path.exists(OLD_LOG):
    old = pd.read_csv(OLD_LOG)
    keep = old[~old["fraction"].isin(df["fraction"].unique())]
    if len(keep):
        print("\n=== first pass, DEV only - label these clearly as dev, unmatched "
              "budgets, 2 seeds ===")
        print(keep.pivot_table(index="fraction", columns="method",
                               values="f1_macro", aggfunc="mean").round(4).to_string())

