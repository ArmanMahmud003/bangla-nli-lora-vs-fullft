# =============================================================================
# STEP 10 - retrain with MATCHED budgets and score the TEST split at train time
# =============================================================================
# Replaces run_experiment(). Every defect this fixes, in order:
#   * both methods get the SAME epoch ceiling and the same early-stopping patience,
#     so neither is cut off while still improving
#   * the held-out test split is scored inside the run, so no checkpoint can ever
#     go missing again and no selection happens on test
#   * per-example test logits are saved, which is what makes McNemar / bootstrap
#     significance tests possible later without paying for GPU twice
#   * trainable parameters, wall-clock, peak GPU memory and saved-model size are
#     recorded - the efficiency claims LoRA is actually about
#   * training checkpoints go to scratch disk, never into PROJECT_DIR; only the best
#     model, curves, predictions and a manifest are kept as results
#   * resume is explicit and gated on an exact settings match, so a rerun is a rerun
#     and only a genuinely interrupted run continues where it stopped
#
# Needs from earlier cells: PROJECT_DIR, tokenized_ds (train/validation/test),
#                           compute_metrics
# Writes: runs/<run_id>.json (source of truth), results_v2.csv (derived),
#         curves_v2/<run_id>.json, preds/<run_id>_test.npz, best_models/<run_id>/
# =============================================================================

import glob, json, math, os, shutil, time
import numpy as np
import pandas as pd
import torch
import transformers, peft
from transformers import (AutoModelForSequenceClassification, EarlyStoppingCallback,
                          Trainer, TrainingArguments, default_data_collator, set_seed)
from peft import LoraConfig, TaskType, get_peft_model

# ---- configuration ----------------------------------------------------------
RESULTS_V2   = f"{PROJECT_DIR}/results_v2.csv"
RUNS_DIR     = f"{PROJECT_DIR}/runs"          # one manifest per run; the CSV is
CURVES_DIR   = f"{PROJECT_DIR}/curves_v2"     # rebuilt from these, so a wiped CSV
PREDS_DIR    = f"{PROJECT_DIR}/preds"         # costs nothing
MODELS_DIR   = f"{PROJECT_DIR}/best_models"
# Training checkpoints live on fast scratch disk and are wiped after each run - they
# must never land inside PROJECT_DIR. Colab: /content/ckpt. Set SCRATCH_CKPT in the
# setup cell to override it (needed when running locally, e.g. on Windows).
SCRATCH_CKPT = globals().get("SCRATCH_CKPT", "/content/ckpt")

BASE_MODEL      = "csebuetnlp/banglabert"
NUM_LABELS      = 3
BATCH_SIZE      = 16                 # same as your first pass, keep it
EVAL_BATCH      = 64
CEILING         = 8                  # SAME for both methods - the whole point
PATIENCE_EPOCHS = None
# None = NO early stopping: every run trains the full ceiling and the epoch with the
# best validation macro-F1 is selected. Deliberate. With patience=2 the pilot showed
# full fine-tuning quitting at epoch 4 after two unlucky dips, while the same run in
# fp32 recovered at epoch 5 with a 0.006 better score - i.e. early stopping was still
# truncating full FT (its dev curve is noisier than LoRA's), which is the defect this
# rewrite exists to remove. Fixed epochs cost ~2x the GPU time at low fractions and
# mixed precision already bought that back. Set it to an int only if you must save
# time, and then re-run every run you already have with force=True.
PRECISION       = globals().get("PRECISION", "fp16")
# Mixed precision: fp16 on T4/V100, bf16 on A100 and RTX 30/40/50-series (bf16 spans
# the same number range as fp32, so it cannot underflow). Set it in the setup cell.
# It must be IDENTICAL for every run in the grid - if you change it, re-run the runs
# you already have with force=True, or the comparison is confounded.
LR       = {"lora": 2e-4, "full_ft": 2e-5}
LORA_CFG = dict(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])

# A full-FT checkpoint is ~443 MB and 25 runs would be ~11 GB of Drive; nothing in
# the thesis needs those weights, because the test predictions and every metric are
# already saved. The size is still MEASURED for the efficiency table - the model is
# written locally, weighed, then deleted. Flip to True if you want the weights.
SAVE_BEST_MODEL = {"lora": True, "full_ft": False}

CHANCE_F1 = 0.40   # three classes, so guessing scores ~0.33 macro-F1. A run at or
                   # below this collapsed (fp16 underflow, or a bad learning rate) -
                   # it is not a result. No manifest is written, so run_grid retries.

# ---- unattended running: power cuts, crashes, and a deadline -----------------
RESUME_INTERRUPTED = globals().get("RESUME_INTERRUPTED", True)
# A run that dies part-way (power cut, blue screen, closed lid) leaves its optimizer
# state, learning-rate schedule, RNG state and log history in SCRATCH_CKPT/<run_id>.
# With this on, the next attempt continues from there instead of throwing away the
# hours already spent - but ONLY if the settings written into that folder at the start
# are identical to the settings now. That gate is the whole difference between this
# and the silent resume_from_checkpoint that broke the first pass: back then a rerun
# quietly continued an old run under new settings and called it a fresh result.
# A resumed run's wall-clock only covers the part after the crash, so the manifest
# records wall_clock_comparable=False and the timing tables drop it. Accuracy is
# unaffected: the schedule is restored, so the model is the one it would have been.

RETRIES = globals().get("RETRIES", 1)
# Extra attempts per run before the grid moves on. A first failure is usually
# transient (a driver hiccup, a full disk, another program grabbing the GPU); the
# retry starts from the surviving checkpoint if there is one.

AUTO_EXPORT_DIR = globals().get("AUTO_EXPORT_DIR", None)
# Set this (e.g. to DOWNLOADS, or a OneDrive/Dropbox folder) and a fresh results zip
# is rewritten after every finished run. If the machine dies at 4 a.m. the zip on
# disk is never more than one run out of date, so nothing has to be re-run to get
# results off the machine. A few MB, overwritten in place.

# Training throughput measured on the first 20 runs, samples/second. Used only to
# estimate how long the remaining grid takes, never for any result.
SPEED_FALLBACK = {"full_ft": 87.5, "lora": 154.3}      # Tesla T4, fp16, batch 16


def check_precision():
    if PRECISION not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"PRECISION must be fp32, fp16 or bf16 - got {PRECISION!r}")
    if PRECISION == "bf16" and torch.cuda.is_available() \
            and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{torch.cuda.get_device_name(0)} cannot do bf16 - "
                           "set PRECISION = 'fp16' on T4/V100 and re-run cell 4.")

# PLACEHOLDER_A


def evals_per_epoch(fraction):
    """One selection point per epoch is plenty when an epoch is short; at 25% of
    the data and above an epoch is thousands of steps, so evaluate twice per epoch
    or the best-checkpoint grid is coarser than the differences you are measuring."""
    return 1 if fraction <= 0.10 else 2


KEEP = ["input_ids", "attention_mask", "token_type_ids", "label"]

def _strip(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in KEEP])

def _dir_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return round(total / 1e6, 2)

def _mkdirs():
    for d in (RUNS_DIR, CURVES_DIR, PREDS_DIR, MODELS_DIR, SCRATCH_CKPT):
        os.makedirs(d, exist_ok=True)

def build_model(method, seed):
    """Returns (model, trainable, total). set_seed first so the classifier head and
    the LoRA A/B matrices are initialised reproducibly."""
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
        print("no run manifests yet")
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["fraction", "seed", "method"])
    df.to_csv(RESULTS_V2, index=False)
    return df

def done_runs():
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{RUNS_DIR}/*.json")}

# ---- moving results between machines that do NOT share a Drive ---------------
# Filenames carry the run id, so two machines never produce the same file for
# different work. That makes a merge safe as long as it refuses to overwrite.
TRANSFER_DIRS = ("runs", "curves_v2", "preds", "best_models")

def import_zip(zip_path, apply=True):
    """Merge another machine's results into this PROJECT_DIR from a zip file.
    Nothing is ever overwritten. A file already here byte-for-byte is skipped; one
    that differs is KEPT as it is and the incoming copy is written beside it as
    <name>.incoming, which is the only real conflict: the same run trained twice.
    The zip can be a whole-folder download from Drive - runs/, curves_v2/, preds/
    and best_models/ are found at any depth inside it, and anything else ignored."""
    import zipfile
    _mkdirs()
    added = same = clash = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            at = [i for i, p in enumerate(parts[:-1]) if p in TRANSFER_DIRS]
            if not at:
                continue
            rel = "/".join(parts[at[-1]:])
            dest = f"{PROJECT_DIR}/{rel}"
            data = z.read(info)
            if os.path.exists(dest):
                if open(dest, "rb").read() == data:
                    same += 1
                else:
                    clash += 1
                    print(f"  [!] {rel} is already here and DIFFERS - local copy kept, "
                          f"incoming saved as {rel}.incoming. That run was trained on "
                          f"both machines; keep one and delete the other.")
                    if apply:
                        open(dest + ".incoming", "wb").write(data)
                continue
            added += 1
            if apply:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(data)
    print(f"import: {added} new file(s), {same} already identical, {clash} conflicting")
    if added and apply:
        rebuild_results_csv()
    return added, same, clash

def newest_zip(folder):
    """Most recently modified .zip in a folder, or None. Used to pick up the file you
    just dropped in Downloads without typing its name."""
    z = sorted((p for p in glob.glob(os.path.join(folder, "*.zip"))), key=os.path.getmtime)
    return z[-1] if z else None

def export_zip(out_dir=None, include_models=False, name=None, quiet=False):
    """Zip this machine's runs/, curves_v2/ and preds/ for the other machine. A few MB.
    Safe to import repeatedly - the merge skips what is already there. The saved LoRA
    adapters are left out unless you ask for them; the analysis does not need them.
    Pass name= to write the same filename every time (used by the automatic backup,
    which overwrites one 'latest' zip instead of leaving one per run)."""
    import zipfile
    out_dir = out_dir or PROJECT_DIR
    dirs = TRANSFER_DIRS if include_models else tuple(d for d in TRANSFER_DIRS
                                                     if d != "best_models")
    tag = (torch.cuda.get_device_name(0).split()[-1] if torch.cuda.is_available() else "cpu")
    out = os.path.join(out_dir, name or
                       f"thesis_runs_{tag}_{time.strftime('%Y%m%d_%H%M')}.zip")
    tmp, n = out + ".part", 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name_ in dirs:
            for p in sorted(glob.glob(f"{PROJECT_DIR}/{name_}/**/*", recursive=True)):
                if os.path.isfile(p):
                    z.write(p, os.path.relpath(p, PROJECT_DIR).replace("\\", "/"))
                    n += 1
    os.replace(tmp, out)      # written aside then moved, so a power cut mid-zip cannot
    if not quiet:             # leave you with a half-written backup under the real name
        print(f"wrote {out}\n  {n} file(s), {round(os.path.getsize(out) / 1e6, 2)} MB "
              f"- copy this to the other machine and import it there")
    return out

# PLACEHOLDER_B

def settings_now(method, fraction, ceiling=None):
    """The settings a run WOULD get right now. Compared against what a saved manifest
    was actually produced with, so a mid-grid change of mind can never quietly leave
    half the grid trained under different rules - that is how the first pass ended up
    with LoRA on 8 epochs and full fine-tuning on 3.
    ceiling=None means 'whatever CEILING says now' - read at call time, not at import
    time, so editing CEILING above and re-running that cell alone takes effect."""
    return {"precision": PRECISION, "epoch_ceiling": CEILING if ceiling is None else ceiling,
            "batch_size": BATCH_SIZE, "patience_epochs": PATIENCE_EPOCHS,
            "learning_rate": LR[method], "evals_per_epoch": evals_per_epoch(fraction)}


def stale_runs(fractions, seeds, methods=("full_ft", "lora"), ceiling=None):
    """[(run_id, {setting: (saved, now)})] for finished runs that no longer match."""
    out = []
    for f in sorted(fractions):
        for s in seeds:
            for m in methods:
                run_id = f"{m}_frac{f}_seed{s}"
                p = f"{RUNS_DIR}/{run_id}.json"
                if not os.path.exists(p):
                    continue
                saved, now = json.load(open(p)), settings_now(m, f, ceiling)
                diff = {k: (saved.get(k), v) for k, v in now.items() if saved.get(k) != v}
                if diff:
                    out.append((run_id, diff))
    return out


def report_stale(fractions, seeds, methods=("full_ft", "lora"), ceiling=None):
    stale = stale_runs(fractions, seeds, methods, ceiling)
    for run_id, diff in stale:
        detail = ", ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in diff.items())
        print(f"  [STALE] {run_id} was trained with {detail}")
    if stale:
        print(f"  {len(stale)} finished run(s) no longer match the current settings. "
              f"Re-run them with force=True, or the comparison is confounded.")


def this_gpu():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def report_split_risk(fractions, seeds, methods=("full_ft", "lora")):
    """Warn BEFORE training when a run would finish a pair whose other half was trained
    on a different machine. The headline number is LoRA minus full fine-tuning at the
    same fraction and seed, so both halves must come off the same GPU or that gap mixes
    hardware with method. Splitting by SEED is safe; splitting by METHOD is not."""
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
        print(f"  [SPLIT RISK] fraction {f:g} seed {s}: {', '.join(todo)} would be trained "
              f"here on {here}, but its partner already ran on {other}. Either finish this "
              f"seed on that machine, or re-run the finished half here with force=True.")
    if risky:
        print(f"  {len(risky)} pair(s) would end up split across machines. Accuracy pools "
              f"across GPUs; a paired gap does not.")
    return risky


# ---- continuing a run that was cut off half-way ------------------------------
class ResumeFailed(RuntimeError):
    """The saved checkpoint could not be read back - almost always because the power
    went during the write. The folder is wiped and the run starts over."""


def _stamp(method, fraction, seed, ceiling, n_train):
    """Everything that would make two attempts at the same run_id different work.
    Written into the checkpoint folder at the start and compared before any resume."""
    s = dict(settings_now(method, fraction, ceiling))
    s.update(run_id=f"{method}_frac{fraction}_seed{seed}", seed=seed, n_train=n_train,
             base_model=BASE_MODEL, lora_cfg=LORA_CFG if method == "lora" else None)
    return s


def _resume_from(run_id, stamp):
    """(checkpoint_dir, step) to continue from, or (None, 0) to start fresh.
    Refuses - and wipes - a checkpoint folder that was written under different
    settings or is unreadable, so a stale folder can never contaminate a result."""
    d = f"{SCRATCH_CKPT}/{run_id}"
    cks = sorted((p for p in glob.glob(f"{d}/checkpoint-*") if os.path.isdir(p)),
                 key=lambda p: int(p.rsplit("-", 1)[1]), reverse=True)
    if not cks:
        return None, 0
    sp = f"{d}/_settings_stamp.json"
    saved = json.load(open(sp)) if os.path.exists(sp) else None
    if saved != stamp:
        why = "has no settings stamp" if saved is None else "was trained under " + ", ".join(
            f"{k}: {saved.get(k)!r} -> {v!r}" for k, v in stamp.items() if saved.get(k) != v)
        print(f"  [FRESH START] the leftover checkpoint for {run_id} {why} - deleting it "
              f"rather than continuing it.")
        shutil.rmtree(d, ignore_errors=True)
        return None, 0
    for p in cks:                      # the newest one may be a half-written folder
        if os.path.exists(f"{p}/trainer_state.json"):
            step = int(p.rsplit("-", 1)[1])
            print(f"  [RESUME] {run_id} was interrupted at step {step} - continuing from "
                  f"that checkpoint. Same settings, so the schedule and optimizer state "
                  f"pick up where they stopped. Wall-clock for this run will not be "
                  f"comparable and is excluded from the timing table.")
            return p, step
    print(f"  [FRESH START] {run_id} has checkpoint folders but none of them finished "
          f"writing - deleting them.")
    shutil.rmtree(d, ignore_errors=True)
    return None, 0


# ---- how long is left, and does it fit before the deadline -------------------
def _full_train_size():
    try:
        return len(tokenized_ds["train"])
    except Exception:
        return 381449              # XNLI-bn train split, for planning before cell 3


def measured_speed(method):
    """Training samples/second for this method on THIS GPU, from runs already finished
    here. Falls back to the T4 numbers until this machine has done one of each."""
    here, vals = this_gpu(), []
    for p in glob.glob(f"{RUNS_DIR}/*.json"):
        r = json.load(open(p))
        if r.get("method") == method and r.get("gpu") == here \
                and r.get("wall_clock_comparable", True) and r.get("train_samples_per_s"):
            vals.append(r["train_samples_per_s"])
    return (float(np.mean(vals)), len(vals)) if vals else (SPEED_FALLBACK[method], 0)


def estimate_minutes(method, fraction, ceiling=None):
    """Rough wall-clock for one run: ceiling passes over the training subset, plus one
    validation pass per evaluation point, plus the single test pass at the end.
    Evaluation has no backward pass, so it runs about three times faster."""
    ceiling = CEILING if ceiling is None else ceiling
    sps, _ = measured_speed(method)
    n_train = int(_full_train_size() * fraction)
    n_eval = 2419 * ceiling * evals_per_epoch(fraction) + 4895
    return (ceiling * n_train / sps + n_eval / (3 * sps)) / 60


def plan(fractions, seeds, methods=("full_ft", "lora"), ceiling=None, deadline=None,
         hours_per_day=24.0):
    """What is left, what it costs, and whether it lands before the deadline.
    deadline is your local time as "YYYY-MM-DD HH:MM". Nothing is trained here."""
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
        print(f"  speed for {m:8s}: {sps:6.1f} samples/s "
              + (f"(measured here over {n} run(s))" if n else
                 "(Tesla T4 fallback - this machine has not finished one yet, so every "
                 "estimate below is provisional)"))
    print(f"\n  {len(todo)} run(s) still to do, {todo.est_min.sum()/60:.1f} GPU-hours; "
          f"{len(df) - len(todo)} already finished")
    if not len(todo):
        print("  nothing left in this list - move on to the next stage, or to the analysis.")
        return df
    print(todo.groupby("fraction")
              .agg(runs=("run_id", "count"), hours=("est_min", lambda x: round(x.sum()/60, 1)))
              .to_string())
    if len(todo):
        finish = time.time() + todo.est_min.sum() * 60 * (24.0 / hours_per_day)
        print(f"\n  start now and the last run ends about "
              f"{time.strftime('%a %d %b %H:%M', time.localtime(finish))}"
              + (f" (running {hours_per_day:g} h/day)" if hours_per_day < 24 else ""))
        if deadline:
            dl = time.mktime(time.strptime(deadline, "%Y-%m-%d %H:%M"))
            left = (dl - time.time()) / 3600
            print(f"  deadline {deadline}: {left:.1f} hours away, "
                  f"{todo.est_min.sum()/60:.1f} needed"
                  + ("  -> it fits" if finish <= dl else
                     "  -> IT DOES NOT FIT. Cut seeds or fractions from the tail, or "
                     "lower the ceiling for BOTH methods."))
            fits, spent = [], 0.0
            for _, r in todo.iterrows():
                spent += r.est_min
                if spent / 60 <= left:
                    fits.append(r.run_id)
            print(f"  runs that fit in the time left: {len(fits)} of {len(todo)}"
                  + (f", last one is {fits[-1]}" if fits else ""))
    return df


def auto_export():
    """Rewrite one always-current zip after every finished run, if AUTO_EXPORT_DIR is
    set. Same filename every time, so it overwrites instead of piling up."""
    if not AUTO_EXPORT_DIR:
        return None
    try:
        os.makedirs(AUTO_EXPORT_DIR, exist_ok=True)
        tag = this_gpu().split()[-1]
        return export_zip(AUTO_EXPORT_DIR, name=f"thesis_runs_{tag}_latest.zip", quiet=True)
    except Exception as e:
        print(f"  [!] auto-export failed ({type(e).__name__}: {e}) - training continues, "
              f"but get the results off this machine by hand.")
        return None


def run_one(method, fraction, seed, ceiling=None, force=False):
    """One run, start to finish: train -> select on dev -> score test -> log. ~1 GPU."""
    ceiling = CEILING if ceiling is None else ceiling
    run_id = f"{method}_frac{fraction}_seed{seed}"
    manifest_path = f"{RUNS_DIR}/{run_id}.json"
    if os.path.exists(manifest_path) and not force:
        saved = json.load(open(manifest_path))
        diff = {k: (saved.get(k), v) for k, v in settings_now(method, fraction, ceiling).items()
                if saved.get(k) != v}
        if diff:
            print(f"[SKIP] {run_id} already done, but under DIFFERENT settings: "
                  + ", ".join(f"{k} {was!r} -> {now!r}" for k, (was, now) in diff.items())
                  + " - re-run it with force=True.")
        else:
            print(f"[SKIP] {run_id} already done.")
        return saved

    _mkdirs()
    check_precision()
    set_seed(seed)
    n_train = int(len(tokenized_ds["train"]) * fraction)
    # identical subset for both methods at a given seed, nested as fraction grows
    train_ds = _strip(tokenized_ds["train"].shuffle(seed=seed).select(range(n_train)))
    dev_ds, test_ds = _strip(tokenized_ds["validation"]), _strip(tokenized_ds["test"])

    steps_per_epoch = max(1, math.ceil(n_train / BATCH_SIZE))
    per_epoch = evals_per_epoch(fraction)
    eval_every = max(1, steps_per_epoch // per_epoch)
    patience = PATIENCE_EPOCHS * per_epoch if PATIENCE_EPOCHS else None
    by_steps = per_epoch > 1

    model, trainable, total = build_model(method, seed)
    print(f"[RUN] {run_id}  n_train={n_train}  ceiling={ceiling} epochs  "
          f"lr={LR[method]}  eval every {eval_every} steps  "
          f"{'patience=%d evals' % patience if patience else 'no early stopping'}  "
          f"trainable={trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    # If this run was cut off before, continue it - but only under identical settings.
    stamp = _stamp(method, fraction, seed, ceiling, n_train)
    ckpt_dir = f"{SCRATCH_CKPT}/{run_id}"
    resume_path, resume_step = _resume_from(run_id, stamp) if RESUME_INTERRUPTED else (None, 0)
    os.makedirs(ckpt_dir, exist_ok=True)
    json.dump(stamp, open(f"{ckpt_dir}/_settings_stamp.json", "w"), indent=2)

    args = TrainingArguments(
        output_dir=f"{SCRATCH_CKPT}/{run_id}",
        num_train_epochs=ceiling,
        learning_rate=LR[method],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH,
        eval_strategy="steps" if by_steps else "epoch",
        save_strategy="steps" if by_steps else "epoch",
        **({"eval_steps": eval_every, "save_steps": eval_every} if by_steps else {}),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        fp16=(PRECISION == "fp16"),
        bf16=(PRECISION == "bf16"),
        logging_steps=max(10, eval_every // 2),
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=dev_ds, compute_metrics=compute_metrics,
                      data_collator=default_data_collator,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
                      if patience else [])

# PLACEHOLDER_C

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        train_out = trainer.train(resume_from_checkpoint=resume_path) if resume_path \
            else trainer.train()
    except Exception as e:
        if not resume_path:
            raise
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        raise ResumeFailed(
            f"[{run_id}] could not continue from {os.path.basename(resume_path)} "
            f"({type(e).__name__}: {e}). That checkpoint was probably being written when "
            f"the power went. It has been deleted; the next attempt starts this run from "
            f"the beginning.") from e
    wall = time.time() - t0
    peak_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) \
        if torch.cuda.is_available() else None

    # ---- what happened, epoch by epoch (saved so a capped run is diagnosable) ---
    curve = [l for l in trainer.state.log_history if "eval_f1_macro" in l]
    if not curve:
        raise RuntimeError(f"[{run_id}] no eval_f1_macro in log_history - "
                           "compute_metrics key mismatch, fix before logging.")
    best = max(curve, key=lambda x: x["eval_f1_macro"])
    json.dump(curve, open(f"{CURVES_DIR}/{run_id}.json", "w"), indent=2)
    if best["eval_f1_macro"] < CHANCE_F1:
        nan_seen = any(isinstance(l.get("loss"), float) and math.isnan(l["loss"])
                       for l in trainer.state.log_history)
        raise RuntimeError(
            f"[{run_id}] best dev macro-F1 {best['eval_f1_macro']:.4f} is at chance "
            f"level{' and the training loss went NaN' if nan_seen else ''} - this run "
            f"collapsed, it is not a result. With fp16 that is usually underflow: use "
            f"bf16 if your GPU supports it. Curve saved for diagnosis; no manifest "
            f"written, so it will be retried.")
    evals_ran = len(curve)
    evals_possible = ceiling * per_epoch
    early_stopped = evals_ran < evals_possible
    ceiling_hit = abs(float(best["epoch"]) - ceiling) < 1e-6

    # ---- dev at the loaded best model: must reproduce the best curve point ------
    dev = trainer.evaluate(dev_ds, metric_key_prefix="dev")
    drift = abs(dev["dev_f1_macro"] - best["eval_f1_macro"])
    if drift > 1e-3:
        print(f"  [!] dev F1 at the restored best model ({dev['dev_f1_macro']:.4f}) "
              f"differs from the best curve point ({best['eval_f1_macro']:.4f}) by "
              f"{drift:.4f} - load_best_model_at_end did not restore what you think.")

    # ---- test: scored ONCE, never used for any choice --------------------------
    pred = trainer.predict(test_ds, metric_key_prefix="test")
    np.savez_compressed(f"{PREDS_DIR}/{run_id}_test.npz",
                        logits=pred.predictions.astype(np.float16),
                        labels=np.asarray(pred.label_ids))

    # ---- the best model itself: weighed always, kept only if asked --------------
    keep_model = SAVE_BEST_MODEL.get(method, True)
    save_dir = f"{MODELS_DIR}/{run_id}" if keep_model \
        else f"{SCRATCH_CKPT}/_size_probe/{run_id}"
    shutil.rmtree(save_dir, ignore_errors=True)
    trainer.save_model(save_dir)
    saved_mb = _dir_mb(save_dir)
    if not keep_model:
        shutil.rmtree(save_dir, ignore_errors=True)   # size recorded, Drive untouched
        shutil.rmtree(f"{MODELS_DIR}/{run_id}", ignore_errors=True)  # stale big copy

# PLACEHOLDER_D

    manifest = {
        "run_id": run_id, "method": method, "fraction": fraction, "seed": seed,
        "n_train": n_train, "epoch_ceiling": ceiling, "best_epoch": float(best["epoch"]),
        "best_step": int(best.get("step", 0)), "evals_ran": evals_ran,
        "evals_possible": evals_possible, "early_stopped": bool(early_stopped),
        "ceiling_hit": bool(ceiling_hit), "patience_evals": patience,
        "patience_epochs": PATIENCE_EPOCHS, "evals_per_epoch": per_epoch,
        "learning_rate": LR[method], "batch_size": BATCH_SIZE, "precision": PRECISION,
        "dev_accuracy": float(dev["dev_accuracy"]), "dev_f1_macro": float(dev["dev_f1_macro"]),
        "dev_f1_from_curve": float(best["eval_f1_macro"]), "dev_reload_drift": float(drift),
        "test_accuracy": float(pred.metrics["test_accuracy"]),
        "test_f1_macro": float(pred.metrics["test_f1_macro"]),
        "trainable_params": int(trainable), "total_params": int(total),
        "pct_trainable": round(100 * trainable / total, 4),
        "train_runtime_s": round(wall, 1),
        "train_samples_per_s": round(train_out.metrics.get("train_samples_per_second", 0), 2),
        "resumed_from_step": int(resume_step),
        "wall_clock_comparable": not resume_step,
        # A resumed run only timed the part after the interruption, so its minutes and
        # samples/second mean nothing. Accuracy is untouched. step11 drops these rows
        # from the speed table and keeps them in every accuracy table.
        "peak_gpu_mem_mb": peak_mb, "saved_model_mb": saved_mb,
        "best_model_kept": bool(keep_model),
        "gpu": this_gpu(),
        "transformers": transformers.__version__, "peft": peft.__version__,
        "torch": torch.__version__,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    rebuild_results_csv()

    print(f"  dev F1 {manifest['dev_f1_macro']:.4f} @ epoch {manifest['best_epoch']:g}"
          f"{'  (early stopped)' if early_stopped else ''}"
          f"{'  (CEILING HIT - budget may still be binding)' if ceiling_hit else ''}"
          f"  ->  TEST acc {manifest['test_accuracy']:.4f} / F1 "
          f"{manifest['test_f1_macro']:.4f}   ["
          f"{'resumed, timing not comparable' if resume_step else f'{wall/60:.1f} min'}, "
          f"{peak_mb} MB peak, model {manifest['saved_model_mb']} MB]")

    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    shutil.rmtree(f"{SCRATCH_CKPT}/{run_id}", ignore_errors=True)   # free local disk
    return manifest


def run_grid(fractions, seeds, methods=("full_ft", "lora"), ceiling=None, force=False,
             retries=None, stop_after_hours=None, deadline=None):
    """Cheapest fractions first, so a disconnect costs you the least. Every finished
    run is saved in PROJECT_DIR before the next one starts, so just re-run this cell
    to continue. Safe to leave running unattended:
      * a run that fails is retried (from its own checkpoint if one survived), and if
        it fails again the grid moves on and the reason is appended to failures.log
      * a fresh results zip is written after every run when AUTO_EXPORT_DIR is set
      * stop_after_hours / deadline stop it LAUNCHING new runs once there is no time
        left, instead of being killed mid-run with nothing to show for the hours
      * Ctrl-C / Interrupt stops after the current run, it does not corrupt anything"""
    retries = RETRIES if retries is None else retries
    todo = [(f, s, m) for f in sorted(fractions) for s in seeds for m in methods]
    print(f"{len(todo)} run(s) queued; {len(done_runs())} already finished in this folder")
    report_stale(fractions, seeds, methods, ceiling)
    report_split_risk(fractions, seeds, methods)
    budget_h = stop_after_hours
    if deadline:
        left = (time.mktime(time.strptime(deadline, "%Y-%m-%d %H:%M")) - time.time()) / 3600
        budget_h = left if budget_h is None else min(budget_h, left)
        print(f"deadline {deadline}: {left:.1f} hours from now")
    if AUTO_EXPORT_DIR:
        print(f"a fresh results zip will be rewritten in {AUTO_EXPORT_DIR} after every run")
    print()

    t_start, ok, failed, skipped = time.time(), 0, [], []
    for f, s, m in todo:
        run_id = f"{m}_frac{f}_seed{s}"
        fresh = not os.path.exists(f"{RUNS_DIR}/{run_id}.json") or force
        if budget_h is not None and fresh:
            spent = (time.time() - t_start) / 3600
            est = estimate_minutes(m, f, ceiling) / 60
            if spent + est > budget_h:
                skipped.append(run_id)
                continue
        for attempt in range(retries + 1):
            try:
                run_one(m, f, s, ceiling=ceiling, force=force)
                if fresh:
                    ok += 1
                    auto_export()
                break
            except KeyboardInterrupt:
                print(f"\n[STOPPED] you interrupted during {run_id}. Nothing is corrupted; "
                      f"re-run this cell and it continues from where it stopped.")
                return rebuild_results_csv()
            except Exception as e:                    # one bad run must not kill the grid
                last = attempt == retries
                print(f"[FAILED] {run_id}: {type(e).__name__}: {e}"
                      + ("" if last else f"  - retrying ({attempt + 2} of {retries + 1})"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if last:
                    failed.append(run_id)
                    with open(f"{PROJECT_DIR}/failures.log", "a") as fh:
                        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{run_id}\t"
                                 f"{type(e).__name__}: {e}\n")

    spent = (time.time() - t_start) / 3600
    print(f"\n{ok} run(s) finished this session in {spent:.1f} h; "
          f"{len(done_runs())} in the folder now")
    if failed:
        print(f"  {len(failed)} still failing after {retries + 1} attempt(s): "
              f"{', '.join(failed)} - see failures.log. No manifest was written for them, "
              f"so re-running this cell tries them again.")
    if skipped:
        print(f"  {len(skipped)} not started because they would not finish inside the "
              f"time budget: {', '.join(skipped)}. Raise stop_after_hours, or drop them.")
    return rebuild_results_csv()


