# Standalone copy of notebook cell 4 (notebooks/Step10_retrain_ColabA100.ipynb).
# The notebook embeds the same code; keep the two in sync.
#
# Two ways to use it:
#   1. notebook: exec() it after the data cell, so PROJECT_DIR, tokenized_ds and
#      compute_metrics come from the notebook's globals.
#   2. command line, no notebook needed:
#        PROJECT_DIR=./thesis_runs SCRATCH_CKPT=/tmp/ckpt \
#          python scripts/step10_train_with_test.py --fractions 0.01 --seeds 42
#      Data is then loaded and tokenized by scripts/step3_data.py.
import os as _os
if "PROJECT_DIR" not in globals():
    PROJECT_DIR = _os.environ.get("PROJECT_DIR", "./thesis_runs")
if "SCRATCH_CKPT" not in globals() and "SCRATCH_CKPT" in _os.environ:
    SCRATCH_CKPT = _os.environ["SCRATCH_CKPT"]
if "PRECISION" not in globals() and "PRECISION" in _os.environ:
    PRECISION = _os.environ["PRECISION"]

# =============================================================================
# STEP 10 v2 - retrain with MATCHED budgets and score the TEST split at train time
# =============================================================================
# Design decisions:
#   * both methods get the SAME epoch ceiling (8) and the same stopping rule
#     (none). Neither is truncated while still improving.
#   * the test split is scored INSIDE the run and its logits saved to a .npz,
#     so no post-hoc checkpoint can vanish and no accidental selection on test
#     can happen.
#   * per-example test logits enable McNemar / paired-bootstrap tests later
#     without paying for GPU again.
#   * trainable params, wall-clock, peak GPU memory, model file size are all
#     recorded - the efficiency claims LoRA is actually about.
#   * training checkpoints live on SCRATCH disk (/content/ckpt); only the final
#     best model, curves, predictions and manifest reach PROJECT_DIR on Drive.
#   * resume is EXPLICIT and gated by a settings-stamp: a rerun is a rerun, and
#     only a genuinely interrupted run continues where it stopped.
#
# [v2] additions over the first-pass trainer:
#   * HeartbeatCallback writes runs_progress/<run_id>.json after every eval
#   * post-save reload verification (bit-exact drift written to manifest)
#   * automatic OOM recovery (retry with smaller batch + grad accum)
#   * disk-space guard before starting every run
# =============================================================================

import glob, json, math, os, shutil, time, gc
import numpy as np
import pandas as pd
import torch
import transformers, peft
from transformers import (
    AutoModelForSequenceClassification, EarlyStoppingCallback,
    Trainer, TrainingArguments, TrainerCallback,
    default_data_collator, set_seed
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# ---- configuration ---------------------------------------------------------
RESULTS_V2   = f"{PROJECT_DIR}/results_v2.csv"
RUNS_DIR     = f"{PROJECT_DIR}/runs"
CURVES_DIR   = f"{PROJECT_DIR}/curves_v2"
PREDS_DIR    = f"{PROJECT_DIR}/preds"
MODELS_DIR   = f"{PROJECT_DIR}/best_models"
PROGRESS_DIR = f"{PROJECT_DIR}/runs_progress"   # [v2] heartbeat manifests
SCRATCH_CKPT = globals().get("SCRATCH_CKPT", "/content/ckpt")

BASE_MODEL      = "csebuetnlp/banglabert"
NUM_LABELS      = 3
BATCH_SIZE      = 16          # identical to the first-pass runs
EVAL_BATCH      = 64
CEILING         = 8           # SAME 8-epoch ceiling for BOTH methods
PATIENCE_EPOCHS = None        # None = NO early stopping (deliberate; see README, 'Why fixed 8 epochs')

# fp16 is the locked protocol precision: all 30 delivered runs (T4 and A100) are
# fp16. bf16 is supported for new experiments, but mixing it into this grid
# confounds the comparison.
PRECISION = globals().get("PRECISION", "fp16")

LR       = {"lora": 2e-4, "full_ft": 2e-5}
LORA_CFG = dict(r=8, lora_alpha=16, lora_dropout=0.1,
                target_modules=["query", "value"])

# Full-FT checkpoints are ~443 MB each; 25 runs would eat 11 GB on Drive.
# The test predictions + manifest already contain everything the thesis needs,
# so full-FT weights are measured (for the efficiency table) then deleted.
# LoRA adapters are ~3.6 MB, so keep those.
SAVE_BEST_MODEL = {"lora": True, "full_ft": False}

# Below this dev macro-F1 the run is treated as a collapse (fp16 underflow,
# NaN loss, bad LR) - no manifest is written, so run_grid retries it.
CHANCE_F1 = 0.40      # 3 classes -> random baseline ~0.33

# ---- unattended running: power cuts, crashes, deadlines --------------------
RESUME_INTERRUPTED = globals().get("RESUME_INTERRUPTED", True)
RETRIES            = globals().get("RETRIES", 1)
AUTO_EXPORT_DIR    = globals().get("AUTO_EXPORT_DIR", None)

# Fallback speed numbers for the time estimator, measured on Tesla T4/bs=16/fp16.
# The estimator replaces these with measurements from THIS GPU once one full run
# of each method has finished here.
SPEED_FALLBACK = {"full_ft": 87.5, "lora": 154.3}

# [v2] Auto-export cadence: dump a fresh zip every N finished runs, not every run.
# Every run is safe but slow; every 3 keeps Drive fresh with much less overhead.
AUTO_EXPORT_EVERY = 3

# ---- one-shot precision/GPU sanity check -----------------------------------
def check_precision():
    if PRECISION not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"PRECISION must be fp32|fp16|bf16 - got {PRECISION!r}")
    if PRECISION == "bf16" and torch.cuda.is_available() \
            and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(0)} cannot do bf16 - "
                           "set PRECISION='fp16' in cell 1 and re-run cell 4.")

def evals_per_epoch(fraction):
    """1 eval/epoch at low fractions (short epochs), 2 at 10%+ (longer epochs)."""
    return 1 if fraction <= 0.10 else 2

# ---- dataset helpers -------------------------------------------------------
KEEP = ["input_ids", "attention_mask", "token_type_ids", "label"]

def _strip(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in KEEP])

def _dir_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return round(total / 1e6, 2)

def _mkdirs():
    for d in (RUNS_DIR, CURVES_DIR, PREDS_DIR, MODELS_DIR,
              PROGRESS_DIR, SCRATCH_CKPT):
        os.makedirs(d, exist_ok=True)

def _free_gb(path):
    return round(shutil.disk_usage(path).free / 1e9, 2)

def this_gpu():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

def build_model(method, seed):
    """(model, trainable, total). set_seed BEFORE construction so classifier
    head init and LoRA A/B matrices are reproducible seed-by-seed."""
    set_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=NUM_LABELS)
    if method == "lora":
        model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, **LORA_CFG))
    else:
        model = base
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total

def rebuild_results_csv():
    """results_v2.csv is DERIVED from runs/*.json - never edit it by hand."""
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{RUNS_DIR}/*.json"))]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["fraction", "seed", "method"])
    df.to_csv(RESULTS_V2, index=False)
    return df

def done_runs():
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(f"{RUNS_DIR}/*.json")}

# ---- moving results between machines (unchanged from v1) -------------------
TRANSFER_DIRS = ("runs", "curves_v2", "preds", "best_models")

def import_zip(zip_path, apply=True):
    """Merge another machine's zip into this PROJECT_DIR. Nothing is overwritten
    - a duplicate byte-identical file is skipped, a differing one is saved as
    .incoming beside the local copy for manual resolution."""
    import zipfile
    _mkdirs()
    added = same = clash = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            parts = info.filename.replace("\\", "/").split("/")
            at = [i for i, p in enumerate(parts[:-1]) if p in TRANSFER_DIRS]
            if not at: continue
            rel = "/".join(parts[at[-1]:])
            dest = f"{PROJECT_DIR}/{rel}"
            data = z.read(info)
            if os.path.exists(dest):
                if open(dest, "rb").read() == data:
                    same += 1
                else:
                    clash += 1
                    print(f"  [!] {rel} differs - local kept, incoming saved as .incoming")
                    if apply:
                        open(dest + ".incoming", "wb").write(data)
                continue
            added += 1
            if apply:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(data)
    print(f"import: {added} new, {same} identical, {clash} conflict")
    if added and apply:
        rebuild_results_csv()
    return added, same, clash

def export_zip(out_dir=None, include_models=False, name=None, quiet=False):
    """Zip runs/, curves_v2/, preds/ for backup. Safe to re-import."""
    import zipfile
    out_dir = out_dir or PROJECT_DIR
    dirs = TRANSFER_DIRS if include_models else tuple(
        d for d in TRANSFER_DIRS if d != "best_models")
    tag = (torch.cuda.get_device_name(0).split()[-1]
           if torch.cuda.is_available() else "cpu")
    out = os.path.join(out_dir, name or
                       f"thesis_runs_{tag}_{time.strftime('%Y%m%d_%H%M')}.zip")
    tmp, n = out + ".part", 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name_ in dirs:
            for p in sorted(glob.glob(f"{PROJECT_DIR}/{name_}/**/*", recursive=True)):
                if os.path.isfile(p):
                    z.write(p, os.path.relpath(p, PROJECT_DIR).replace("\\", "/"))
                    n += 1
    os.replace(tmp, out)   # atomic move so a crash mid-zip cannot leave junk
    if not quiet:
        print(f"wrote {out}  ({n} files, {round(os.path.getsize(out)/1e6, 2)} MB)")
    return out

# ---- settings stamp (guarantees fair comparison) ---------------------------
def settings_now(method, fraction, ceiling=None):
    return {"precision": PRECISION,
            "epoch_ceiling": CEILING if ceiling is None else ceiling,
            "batch_size": BATCH_SIZE, "patience_epochs": PATIENCE_EPOCHS,
            "learning_rate": LR[method], "evals_per_epoch": evals_per_epoch(fraction)}

def stale_runs(fractions, seeds, methods=("full_ft", "lora"), ceiling=None):
    out = []
    for f in sorted(fractions):
        for s in seeds:
            for m in methods:
                run_id = f"{m}_frac{f}_seed{s}"
                p = f"{RUNS_DIR}/{run_id}.json"
                if not os.path.exists(p): continue
                saved, now = json.load(open(p)), settings_now(m, f, ceiling)
                diff = {k: (saved.get(k), v) for k, v in now.items()
                        if saved.get(k) != v}
                if diff: out.append((run_id, diff))
    return out

def report_stale(fractions, seeds, methods=("full_ft", "lora"), ceiling=None):
    stale = stale_runs(fractions, seeds, methods, ceiling)
    for run_id, diff in stale:
        detail = ", ".join(f"{k}: {w!r} -> {n!r}" for k, (w, n) in diff.items())
        print(f"  [STALE] {run_id} was trained with {detail}")
    if stale:
        print(f"  {len(stale)} finished run(s) no longer match. Re-run with force=True.")

def report_split_risk(fractions, seeds, methods=("full_ft", "lora")):
    """Warn BEFORE training when a pair (same fraction+seed, both methods) would
    end up split across two GPUs. The per-seed gap must come from one GPU."""
    here, risky = this_gpu(), []
    for f in sorted(fractions):
        for s in seeds:
            saved = {}
            for m in methods:
                p = f"{RUNS_DIR}/{m}_frac{f}_seed{s}.json"
                if os.path.exists(p):
                    saved[m] = json.load(open(p)).get("gpu", "?")
            todo = [m for m in methods if m not in saved]
            other = {g for g in saved.values() if g != here}
            if todo and other:
                risky.append((f, s, todo, sorted(other)))
    for f, s, todo, other in risky:
        print(f"  [SPLIT RISK] fraction {f:g} seed {s}: {todo} would run here "
              f"on {here}, partner already on {other}")
    return risky

# ---- resume-from-checkpoint (unchanged from v1 core; robust) ---------------
class ResumeFailed(RuntimeError):
    """Saved checkpoint could not be read back - usually power cut mid-write."""

def _stamp(method, fraction, seed, ceiling, n_train):
    s = dict(settings_now(method, fraction, ceiling))
    s.update(run_id=f"{method}_frac{fraction}_seed{seed}", seed=seed, n_train=n_train,
             base_model=BASE_MODEL,
             lora_cfg=LORA_CFG if method == "lora" else None)
    return s

def _resume_from(run_id, stamp):
    d = f"{SCRATCH_CKPT}/{run_id}"
    cks = sorted((p for p in glob.glob(f"{d}/checkpoint-*") if os.path.isdir(p)),
                 key=lambda p: int(p.rsplit("-", 1)[1]), reverse=True)
    if not cks: return None, 0
    sp = f"{d}/_settings_stamp.json"
    saved = json.load(open(sp)) if os.path.exists(sp) else None
    if saved != stamp:
        why = "has no settings stamp" if saved is None else "was trained under " + \
              ", ".join(f"{k}: {saved.get(k)!r} -> {v!r}"
                        for k, v in stamp.items() if saved.get(k) != v)
        print(f"  [FRESH START] leftover checkpoint for {run_id} {why} - deleting.")
        shutil.rmtree(d, ignore_errors=True)
        return None, 0
    for p in cks:
        if os.path.exists(f"{p}/trainer_state.json"):
            step = int(p.rsplit("-", 1)[1])
            print(f"  [RESUME] {run_id} continuing from step {step}. Wall-clock "
                  f"for this run is not comparable and is excluded from timing.")
            return p, step
    print(f"  [FRESH START] {run_id} has partial checkpoints - deleting them.")
    shutil.rmtree(d, ignore_errors=True)
    return None, 0

# ---- [v2] HeartbeatCallback: mid-run progress visible from Drive ------------
class HeartbeatCallback(TrainerCallback):
    """Writes a tiny JSON after every evaluation into runs_progress/<run_id>.json.
    Contains current step/epoch, best-so-far F1 and clock time. If Colab kills
    the tab, the last heartbeat on Drive shows how far the run got."""
    def __init__(self, run_id, method, fraction, seed):
        self.run_id, self.method = run_id, method
        self.fraction, self.seed = fraction, seed
        self.t0 = time.time()
        self.best = -1.0
        self.path = f"{PROGRESS_DIR}/{run_id}.json"

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None: return
        f1 = metrics.get("eval_f1_macro", -1.0)
        if f1 > self.best: self.best = f1
        json.dump({
            "run_id": self.run_id, "method": self.method,
            "fraction": self.fraction, "seed": self.seed,
            "step": int(state.global_step), "epoch": float(state.epoch or 0),
            "elapsed_min": round((time.time() - self.t0) / 60, 2),
            "last_eval_f1": float(f1), "best_f1_so_far": float(self.best),
            "gpu": this_gpu(), "updated_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, open(self.path, "w"), indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        if os.path.exists(self.path):
            os.remove(self.path)   # tidy: heartbeat gone once the run is done

# ---- speed & planning (unchanged core) -------------------------------------
def _full_train_size():
    try: return len(tokenized_ds["train"])
    except NameError: return 381449   # XNLI-bn train, before data is loaded

def measured_speed(method):
    here, vals = this_gpu(), []
    for p in glob.glob(f"{RUNS_DIR}/*.json"):
        r = json.load(open(p))
        if r.get("method") == method and r.get("gpu") == here \
                and r.get("wall_clock_comparable", True) \
                and r.get("train_samples_per_s"):
            vals.append(r["train_samples_per_s"])
    return (float(np.mean(vals)), len(vals)) if vals \
        else (SPEED_FALLBACK[method], 0)

def estimate_minutes(method, fraction, ceiling=None):
    ceiling = CEILING if ceiling is None else ceiling
    sps, _ = measured_speed(method)
    n_train = int(_full_train_size() * fraction)
    n_eval = 2419 * ceiling * evals_per_epoch(fraction) + 4895
    return (ceiling * n_train / sps + n_eval / (3 * sps)) / 60

def plan(fractions, seeds, methods=("full_ft", "lora"), ceiling=None,
         deadline=None, hours_per_day=24.0):
    rows = []
    for f in sorted(fractions):
        for s in seeds:
            for m in methods:
                run_id = f"{m}_frac{f}_seed{s}"
                done = os.path.exists(f"{RUNS_DIR}/{run_id}.json")
                mins = 0.0 if done else estimate_minutes(m, f, ceiling)
                rows.append(dict(run_id=run_id, fraction=f, seed=s, method=m,
                                 status="done" if done else "todo",
                                 est_min=round(mins, 1)))
    df = pd.DataFrame(rows)
    df["cum_h"] = (df["est_min"].cumsum() / 60).round(2)
    todo = df[df.status == "todo"]
    for m in methods:
        sps, n = measured_speed(m)
        print(f"  {m:8s}: {sps:6.1f} samples/s "
              + (f"({n} run(s) here)" if n else "(T4 fallback)"))
    print(f"\n  {len(todo)} to do, {todo.est_min.sum()/60:.1f} GPU-h; "
          f"{len(df)-len(todo)} done")
    return df

def auto_export():
    if not AUTO_EXPORT_DIR: return None
    try:
        os.makedirs(AUTO_EXPORT_DIR, exist_ok=True)
        tag = this_gpu().split()[-1]
        return export_zip(AUTO_EXPORT_DIR,
                          name=f"thesis_runs_{tag}_latest.zip", quiet=True)
    except Exception as e:
        print(f"  [!] auto-export failed ({type(e).__name__}: {e})")
        return None

# ---- [v2] post-save reload verification ------------------------------------
def _reload_and_verify(save_dir, method, test_ds, expected_logits):
    """Reload the saved model from disk and re-score the test set.
    Compare against the in-memory predictions. Returns max abs logit drift.
    A drift of 0.000000 (bit-exact) means load_best_model_at_end + trainer.save_model
    round-tripped cleanly; anything above 1e-3 flags a serialization bug."""
    try:
        if method == "lora":
            base = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL, num_labels=NUM_LABELS)
            reloaded = PeftModel.from_pretrained(base, save_dir)
        else:
            reloaded = AutoModelForSequenceClassification.from_pretrained(save_dir)
        reloaded.eval()
        if torch.cuda.is_available(): reloaded = reloaded.cuda()
        # Run through the Trainer's predict path for consistency
        v_trainer = Trainer(model=reloaded, args=TrainingArguments(
            output_dir="/tmp/_verify", per_device_eval_batch_size=EVAL_BATCH,
            report_to="none",
            fp16=(PRECISION == "fp16"), bf16=(PRECISION == "bf16")),
            data_collator=default_data_collator)
        v_pred = v_trainer.predict(test_ds)
        drift = float(np.max(np.abs(v_pred.predictions.astype(np.float32)
                                    - expected_logits.astype(np.float32))))
        del reloaded, v_trainer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return drift, True
    except Exception as e:
        print(f"  [!] reload verification failed: {type(e).__name__}: {e}")
        return None, False

# ---- the actual run -------------------------------------------------------
def run_one(method, fraction, seed, ceiling=None, force=False,
            _oom_batch=None, _oom_accum=1):
    """One run start to finish. train -> select on dev -> score test ->
       reload+verify -> log. About 5-15 min on A100 at fractions 0.01-0.05.
       _oom_batch / _oom_accum are used only by the OOM auto-retry path."""
    ceiling = CEILING if ceiling is None else ceiling
    run_id = f"{method}_frac{fraction}_seed{seed}"
    manifest_path = f"{RUNS_DIR}/{run_id}.json"

    # Skip if already finished, unless settings changed since then.
    if os.path.exists(manifest_path) and not force:
        saved = json.load(open(manifest_path))
        diff = {k: (saved.get(k), v)
                for k, v in settings_now(method, fraction, ceiling).items()
                if saved.get(k) != v}
        if diff:
            print(f"[SKIP] {run_id} already done under DIFFERENT settings: "
                  + ", ".join(f"{k} {w!r}->{n!r}" for k, (w, n) in diff.items())
                  + " - re-run with force=True.")
        else:
            print(f"[SKIP] {run_id} already done.")
        return saved

    _mkdirs()
    check_precision()

    # [v2] disk-space guard: don't start a run that will fail on save
    if _free_gb(SCRATCH_CKPT) < 3:
        raise RuntimeError(f"scratch disk has {_free_gb(SCRATCH_CKPT)} GB free "
                           f"- need at least 3. Free /content and re-run.")
    set_seed(seed)

    n_train = int(len(tokenized_ds["train"]) * fraction)
    # Identical subset for both methods at a given seed; nested as fraction grows.
    train_ds = _strip(tokenized_ds["train"].shuffle(seed=seed).select(range(n_train)))
    dev_ds   = _strip(tokenized_ds["validation"])
    test_ds  = _strip(tokenized_ds["test"])

    effective_batch = _oom_batch or BATCH_SIZE
    steps_per_epoch = max(1, math.ceil(n_train / effective_batch))
    per_epoch = evals_per_epoch(fraction)
    eval_every = max(1, steps_per_epoch // per_epoch)
    patience = PATIENCE_EPOCHS * per_epoch if PATIENCE_EPOCHS else None
    by_steps = per_epoch > 1

    model, trainable, total = build_model(method, seed)
    print(f"[RUN] {run_id}  n_train={n_train}  ceiling={ceiling} epochs  "
          f"lr={LR[method]}  eval every {eval_every} steps  "
          f"{'patience=%d evals' % patience if patience else 'no early stopping'}  "
          f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")
    if _oom_batch:
        print(f"  [OOM RETRY] using batch={_oom_batch}, grad_accum={_oom_accum} "
              f"(effective batch preserved at {BATCH_SIZE})")

    # Continue from a genuine interruption if one exists and matches settings.
    stamp = _stamp(method, fraction, seed, ceiling, n_train)
    ckpt_dir = f"{SCRATCH_CKPT}/{run_id}"
    resume_path, resume_step = _resume_from(run_id, stamp) \
        if RESUME_INTERRUPTED else (None, 0)
    os.makedirs(ckpt_dir, exist_ok=True)
    json.dump(stamp, open(f"{ckpt_dir}/_settings_stamp.json", "w"), indent=2)

    args = TrainingArguments(
        output_dir=f"{SCRATCH_CKPT}/{run_id}",
        num_train_epochs=ceiling,
        learning_rate=LR[method],
        per_device_train_batch_size=effective_batch,
        per_device_eval_batch_size=EVAL_BATCH,
        gradient_accumulation_steps=_oom_accum,
        eval_strategy="steps" if by_steps else "epoch",
        save_strategy="steps" if by_steps else "epoch",
        **({"eval_steps": eval_every, "save_steps": eval_every} if by_steps else {}),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed, data_seed=seed,
        fp16=(PRECISION == "fp16"),
        bf16=(PRECISION == "bf16"),
        logging_steps=max(10, eval_every // 2),
        report_to="none",
    )

    callbacks = [HeartbeatCallback(run_id, method, fraction, seed)]
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=dev_ds,
                      compute_metrics=compute_metrics,
                      data_collator=default_data_collator,
                      callbacks=callbacks)

    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        train_out = trainer.train(resume_from_checkpoint=resume_path) \
            if resume_path else trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        # [v2] OOM auto-retry: halve the batch, double grad_accum, restart clean
        if _oom_batch and _oom_batch <= 4:
            raise
        new_batch = (_oom_batch or BATCH_SIZE) // 2
        new_accum = _oom_accum * 2
        print(f"  [OOM] retrying with batch={new_batch}, grad_accum={new_accum}")
        del model, trainer
        gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        return run_one(method, fraction, seed, ceiling, force=True,
                       _oom_batch=new_batch, _oom_accum=new_accum)
    except Exception as e:
        if not resume_path:
            raise
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        raise ResumeFailed(
            f"[{run_id}] could not continue from {os.path.basename(resume_path)}: "
            f"{type(e).__name__}: {e}. Checkpoint deleted; next attempt is fresh."
        ) from e

    wall = time.time() - t0
    peak_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) \
        if torch.cuda.is_available() else None

    # ---- what happened, epoch by epoch ----------------------------------
    curve = [l for l in trainer.state.log_history if "eval_f1_macro" in l]
    if not curve:
        raise RuntimeError(f"[{run_id}] no eval_f1_macro in log_history")
    best = max(curve, key=lambda x: x["eval_f1_macro"])
    json.dump(curve, open(f"{CURVES_DIR}/{run_id}.json", "w"), indent=2)
    if best["eval_f1_macro"] < CHANCE_F1:
        nan_seen = any(isinstance(l.get("loss"), float) and math.isnan(l["loss"])
                       for l in trainer.state.log_history)
        raise RuntimeError(
            f"[{run_id}] best dev macro-F1 {best['eval_f1_macro']:.4f} at chance"
            f"{' + NaN loss' if nan_seen else ''} - collapsed run, no manifest.")
    evals_ran = len(curve)
    evals_possible = ceiling * per_epoch
    early_stopped = evals_ran < evals_possible
    ceiling_hit = abs(float(best["epoch"]) - ceiling) < 1e-6

    # ---- dev at the loaded best model: must match the best curve point --
    dev = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    drift = abs(dev["dev_f1_macro"] - best["eval_f1_macro"])
    if drift > 1e-3:
        print(f"  [!] dev F1 drift on reload: {drift:.4f} - "
              f"load_best_model_at_end did not restore the best checkpoint")

    # ---- test: scored ONCE, never used for selection ---------------------
    pred = trainer.predict(test_ds, metric_key_prefix="test")
    np.savez_compressed(f"{PREDS_DIR}/{run_id}_test.npz",
                        logits=pred.predictions.astype(np.float16),
                        labels=np.asarray(pred.label_ids))

    # ---- the best model: weighed always, kept only if asked --------------
    keep_model = SAVE_BEST_MODEL.get(method, True)
    save_dir = f"{MODELS_DIR}/{run_id}" if keep_model \
        else f"{SCRATCH_CKPT}/_size_probe/{run_id}"
    shutil.rmtree(save_dir, ignore_errors=True)
    trainer.save_model(save_dir)
    saved_mb = _dir_mb(save_dir)

    # [v2] reload verification: load from disk, re-score test, compare logits
    reload_drift, reload_ok = _reload_and_verify(
        save_dir, method, test_ds, pred.predictions)

    if not keep_model:
        shutil.rmtree(save_dir, ignore_errors=True)
        shutil.rmtree(f"{MODELS_DIR}/{run_id}", ignore_errors=True)

    manifest = {
        "run_id": run_id, "method": method, "fraction": fraction, "seed": seed,
        "n_train": n_train, "epoch_ceiling": ceiling,
        "best_epoch": float(best["epoch"]),
        "best_step": int(best.get("step", 0)),
        "evals_ran": evals_ran, "evals_possible": evals_possible,
        "early_stopped": bool(early_stopped), "ceiling_hit": bool(ceiling_hit),
        "patience_evals": patience, "patience_epochs": PATIENCE_EPOCHS,
        "evals_per_epoch": per_epoch,
        "learning_rate": LR[method], "batch_size": BATCH_SIZE,
        "effective_batch": effective_batch * _oom_accum,
        "grad_accum_steps": _oom_accum,
        "precision": PRECISION,
        "dev_accuracy": float(dev["dev_accuracy"]),
        "dev_f1_macro": float(dev["dev_f1_macro"]),
        "dev_f1_from_curve": float(best["eval_f1_macro"]),
        "dev_reload_drift": float(drift),
        "test_accuracy": float(pred.metrics["test_accuracy"]),
        "test_f1_macro": float(pred.metrics["test_f1_macro"]),
        "trainable_params": int(trainable), "total_params": int(total),
        "pct_trainable": round(100 * trainable / total, 4),
        "train_runtime_s": round(wall, 1),
        "train_samples_per_s": round(
            train_out.metrics.get("train_samples_per_second", 0), 2),
        "resumed_from_step": int(resume_step),
        "wall_clock_comparable": not resume_step,
        "peak_gpu_mem_mb": peak_mb, "saved_model_mb": saved_mb,
        "best_model_kept": bool(keep_model),
        "reload_verified": bool(reload_ok),
        "reload_max_logit_drift": (round(reload_drift, 8)
                                   if reload_drift is not None else None),
        "gpu": this_gpu(),
        "transformers": transformers.__version__, "peft": peft.__version__,
        "torch": torch.__version__,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    rebuild_results_csv()

    print(f"  dev F1 {manifest['dev_f1_macro']:.4f} @ epoch {manifest['best_epoch']:g}"
          f"{'  (early stopped)' if early_stopped else ''}"
          f"{'  (CEILING HIT)' if ceiling_hit else ''}"
          f"  ->  TEST acc {manifest['test_accuracy']:.4f} / "
          f"F1 {manifest['test_f1_macro']:.4f}   "
          f"[{wall/60:.1f} min, {peak_mb} MB peak, model {saved_mb} MB, "
          f"reload drift {manifest['reload_max_logit_drift']}]")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    shutil.rmtree(f"{SCRATCH_CKPT}/{run_id}", ignore_errors=True)
    return manifest


def run_grid(fractions, seeds, methods=("full_ft", "lora"), ceiling=None,
             force=False, retries=None, stop_after_hours=None, deadline=None):
    """Cheapest fractions first. Every finished run is saved to Drive before the
    next starts - re-running this cell after a disconnect just continues.
    Safe to leave unattended: transient failures are retried, auto-export writes
    a fresh backup zip periodically, and Ctrl-C stops between runs cleanly."""
    retries = RETRIES if retries is None else retries
    todo = [(f, s, m) for f in sorted(fractions) for s in seeds for m in methods]
    print(f"{len(todo)} run(s) queued; {len(done_runs())} already finished")
    report_stale(fractions, seeds, methods, ceiling)
    report_split_risk(fractions, seeds, methods)
    budget_h = stop_after_hours
    if deadline:
        left = (time.mktime(time.strptime(deadline, "%Y-%m-%d %H:%M"))
                - time.time()) / 3600
        budget_h = left if budget_h is None else min(budget_h, left)
        print(f"deadline {deadline}: {left:.1f} hours from now")
    if AUTO_EXPORT_DIR:
        print(f"auto-export every {AUTO_EXPORT_EVERY} run(s) -> {AUTO_EXPORT_DIR}")
    print()

    t_start, ok, failed, skipped = time.time(), 0, [], []
    for i, (f, s, m) in enumerate(todo):
        run_id = f"{m}_frac{f}_seed{s}"
        fresh = not os.path.exists(f"{RUNS_DIR}/{run_id}.json") or force
        if budget_h is not None and fresh:
            spent = (time.time() - t_start) / 3600
            est = estimate_minutes(m, f, ceiling) / 60
            if spent + est > budget_h:
                skipped.append(run_id); continue
        for attempt in range(retries + 1):
            try:
                run_one(m, f, s, ceiling=ceiling, force=force)
                if fresh:
                    ok += 1
                    if ok % AUTO_EXPORT_EVERY == 0:
                        auto_export()
                break
            except KeyboardInterrupt:
                print(f"\n[STOPPED] interrupted during {run_id}. Re-run this "
                      f"cell and it continues cleanly.")
                auto_export()
                return rebuild_results_csv()
            except Exception as e:
                last = attempt == retries
                print(f"[FAILED] {run_id}: {type(e).__name__}: {e}"
                      + ("" if last else f"  - retry {attempt+2}/{retries+1}"))
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if last:
                    failed.append(run_id)
                    with open(f"{PROJECT_DIR}/failures.log", "a") as fh:
                        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
                                 f"{run_id}\t{type(e).__name__}: {e}\n")

    # Final export at end of grid
    auto_export()
    spent = (time.time() - t_start) / 3600
    print(f"\n{ok} run(s) finished this session in {spent:.1f} h; "
          f"{len(done_runs())} total in folder")
    if failed:
        print(f"  {len(failed)} failed after retries: {', '.join(failed)}")
    if skipped:
        print(f"  {len(skipped)} skipped (out of time): {', '.join(skipped)}")
    return rebuild_results_csv()


def load_data():
    """Fill tokenized_ds and compute_metrics from step3_data when they are not
    already defined (i.e. outside the notebook)."""
    global tokenized_ds, compute_metrics
    if "tokenized_ds" not in globals() or "compute_metrics" not in globals():
        from step3_data import tokenized_ds, compute_metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the LoRA vs full-FT grid.")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 21, 33])
    ap.add_argument("--methods", nargs="+", default=["full_ft", "lora"],
                    choices=["full_ft", "lora"])
    ap.add_argument("--force", action="store_true",
                    help="re-train runs that already have a manifest")
    ap.add_argument("--plan", action="store_true",
                    help="print the time estimate and exit without training")
    a = ap.parse_args()
    if a.plan:
        plan(a.fractions, a.seeds, tuple(a.methods))
    else:
        load_data()
        print(run_grid(a.fractions, a.seeds, tuple(a.methods), force=a.force))
