# Bangla NLI — LoRA vs Full Fine-tuning

**BSc thesis code and results.** A matched-budget comparison of LoRA (Low-Rank
Adaptation) against full fine-tuning of [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert)
on the [XNLI-BN](https://huggingface.co/datasets/csebuetnlp/xnli_bn) 3-class natural language
inference task, run at 1%, 5%, and 10% of the training data with five random seeds per cell
(30 runs).

> **Author:** Arman Mahmud · **Base model:** `csebuetnlp/banglabert` · **Task:** 3-class NLI
> (entailment / neutral / contradiction) · **Dataset:** `csebuetnlp/xnli_bn`

---

## Headline result

Across all three data sizes tested, **LoRA and full fine-tuning are statistically
equivalent** on held-out test macro-F1 (gaps of +0.0010 to +0.0024, TOST equivalence
confirmed at all three fractions). The largest gap is **smaller than the ±0.0068
reproducibility floor** measured by re-running identical configurations — the method
difference is below the noise of re-running the same experiment — and every 95% CI is
narrower than the ±0.0126 binomial noise of a single evaluation on 4,895 test items.

LoRA reaches this parity while training **0.80% of the parameters** (887,811 of
110,619,651), shipping a **124× smaller adapter** (3.57 MB vs 442.51 MB), and cutting
peak GPU memory by **25% on a T4** and **42% on an A100**.

**Retracted claim: LoRA is not faster to train on server-class hardware.** It trains
1.5–1.9× faster than full fine-tuning on a T4, but on an A100 it is actually ~1.07×
*slower* wall-clock, and 1.7–2.7× slower to reach the checkpoint you keep. The T4 speed-up
is a low-end-GPU artefact (the fp32 optimizer step dominates step time on weak GPUs, not
strong ones) — this corrects an assumption commonly made about LoRA.

Three additional findings from the same 30 runs:

- **At 1% data, full fine-tuning is severely overconfident** (ECE 0.245 vs 0.117 for LoRA,
  5/5 seeds agree, 95% CI on the gap [−0.140, −0.114]). The advantage is **not monotone**:
  it shrinks by 5% (p = 0.46) and partly returns at 10% (ΔECE −0.067, p = 0.087) — reported
  as three measured points, not a trend. A residual, statistically significant LoRA
  advantage survives even after post-hoc temperature scaling. Traced to full fine-tuning's
  best-epoch selection scattering 2–4× more across seeds than LoRA's — a checkpoint-
  selection artefact of this protocol, not claimed as an intrinsic property of LoRA.
- **The two methods converge on completely different schedules at 5%**: full fine-tuning
  peaks by epoch 3 and then decays (losing 0.0103 macro-F1 by epoch 8); LoRA peaks at
  epoch 6 or later — no overlap across seeds.
- **Equal accuracy hides different models**: the two methods agree on only 84% (1% data) to
  87% (5% data) of individual test predictions — the same aggregate score from two
  measurably different learned functions.

See [`docs/THESIS_MASTER_CONTEXT_v2.md`](docs/THESIS_MASTER_CONTEXT_v2.md) for the full write-up
with every number, method, and caveat.

---

## Experiment setup

| Component | Setting |
|---|---|
| Base model | `csebuetnlp/banglabert` (110,619,651 parameters) |
| Task | 3-class NLI (entailment / neutral / contradiction) |
| Dataset | `csebuetnlp/xnli_bn`, parquet revision |
| Splits | 381,449 train / 2,419 validation / 4,895 test |
| Preprocessing | `csebuetnlp/normalizer` on both sentences, `max_length=128`, pad to max |
| Full fine-tuning | all parameters, learning rate `2e-5` |
| LoRA | `r=8`, `alpha=16`, `dropout=0.1`, adapters on `query` and `value`, LR `2e-4` |
| Precision | fp16 everywhere (bf16 on A100 notebook variant) |
| Batch size | 16 train / 64 eval |
| Epoch budget | **fixed 8 epochs for both methods, no early stopping** |
| Selection | best epoch chosen on validation macro-F1; test scored **once** per run |
| Data subsets | `train.shuffle(seed=seed).select(range(n))` — nested across fractions |

**Why fixed 8 epochs, no early stopping.** In a pilot, patience-2 cut full fine-tuning off at
epoch 4 with its best at epoch 2, while LoRA ran to convergence. Removing early stopping
recovered +0.0161 macro-F1 for full FT alone. A shared ceiling is the honest budget.

**Why select on macro-F1, not validation loss.** At 5%, full-FT validation loss climbs from
0.77 to 1.54 while its macro-F1 still peaks mid-run — loss selection would throw the best
model away.

---

## Repo layout

```
bangla-nli-lora-vs-fullft/
├── README.md                                  ← you are here
├── LICENSE                                    ← MIT
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── Step10_retrain_ColabA100.ipynb         ← end-to-end trainer (A100 / bf16)
├── scripts/
│   ├── step3_data.py                          ← data loading, normalisation, tokenisation
│   ├── step10_train_with_test.py              ← the real trainer (shared with the notebook)
│   └── step11_analysis.py                     ← test tables, significance tests, figures
├── docs/
│   └── THESIS_MASTER_CONTEXT_v2.md             ← full write-up + every number
└── step10_artifacts/                          ← the 30-run evidence base
    ├── results_v2.csv                         ← one row per run, source of the tables below
    ├── runs/<id>.json                         ← one manifest per run
    ├── curves_v2/<id>.json                    ← per-epoch dev curves
    └── preds/<id>_test.npz                    ← per-example test logits
```

The notebook is the current, correct pipeline: matched 8-epoch ceiling, no early stopping,
test scored **inside** each run right after the dev-best checkpoint is restored, per-example
test logits saved, efficiency logged (trainable parameters, wall-clock, peak GPU memory,
artefact size, library versions, GPU name). `scripts/step10_train_with_test.py` and
`scripts/step3_data.py` are the exact modules the notebook runs; `scripts/step11_analysis.py`
reads `step10_artifacts/runs/*.json` + `preds/*.npz` and produces the tables and figures below.

The 3070-specific orchestration (`run_all_3070.py`, `check_progress.py`), the Colab-recovery
utility `merge_zip.py`, and the superseded first-pass eval (`step9_test_eval.py`,
`step9b_equal_budget.py`) were not part of producing the delivered 30 runs and are not
included here.

---

## Reproducing a run

1. Open `notebooks/Step10_retrain_ColabA100.ipynb` in Google Colab on an A100 (bf16) — or
   a T4 (fp16) with the precision cell adjusted. The notebook mounts Drive, installs
   dependencies, and requires **a runtime restart** after the deps cell.
2. Set `PROJECT_DIR` (default `/content/drive/MyDrive/thesis_bangla_nli`). Outputs land in
   `runs/<id>.json`, `curves_v2/<id>.json`, and `preds/<id>_test.npz`.
3. Run all cells. One `(method, fraction, seed)` cell produces one manifest.

Dataset caveats: the 30 runs were split across a T4 and an A100 — 10% data ran entirely on
the A100, 1% mostly on the T4, 5% split across both (disclosed; paired within-method
comparisons are unaffected since every (fraction, seed) pair shares a GPU). Wall-clock and
peak-memory columns **cannot be pooled across different GPUs** — see the retracted
speed claim above for why this matters.

---

## Results snapshot (30 runs, 5 seeds per cell)

**Accuracy — bounded null, equivalence confirmed at all three fractions:**

| Fraction | Full FT (test macro-F1) | LoRA (test macro-F1) | Gap (LoRA − full) | 95% CI | Paired p |
|---|---|---|---|---|---|
| 0.01 (3,814 examples)  | 0.7172 (sd 0.0034) | 0.7183 (sd 0.0080) | +0.0010 | [−0.0054, +0.0074] | 0.685 |
| 0.05 (19,072 examples) | 0.7683 (sd 0.0072) | 0.7706 (sd 0.0019) | +0.0024 | [−0.0076, +0.0123] | 0.547 |
| 0.10 (38,144 examples) | 0.7842 (sd 0.0027) | 0.7851 (sd 0.0044) | +0.0010 | [−0.0062, +0.0082] | 0.728 |

Every gap is smaller than the ±0.0068 reproducibility floor. Only 1 of 15 per-seed McNemar
tests reached p < 0.05, and it did not survive Holm correction — no reliable per-seed
accuracy claim.

**Efficiency (parameters and artefact size hold regardless of hardware):**

| Metric | Full fine-tuning | LoRA | Ratio |
|---|---|---|---|
| Trainable parameters | 110,619,651 | 887,811 | **0.80%** → 125× fewer |
| Artefact on disk | 442.51 MB | 3.57 MB | **124× smaller** |
| Peak GPU memory (A100) | 2,711 MB | 1,568 MB | **−42%** |
| Peak GPU memory (T4) | 2,419 MB | 1,804 MB | **−25%** |
| Training throughput, T4 (samples/s) | 89.5 | 156.2 | LoRA **1.7× faster** |
| Training throughput, A100 (samples/s) | 268.4 | 253.1 | LoRA **~6% slower** |

The T4/A100 throughput reversal is why the "LoRA trains faster" claim above is retracted
rather than reported as a universal efficiency win.

**Calibration — non-monotone, strongest at 1%:**

| Fraction | Full FT ECE | LoRA ECE | ΔECE (LoRA − full) | Paired p |
|---|---|---|---|---|
| 0.01 | 0.245 (sd 0.009) | 0.117 (sd 0.009) | **−0.128** | 0.00001 |
| 0.05 | 0.102 (sd 0.070) | 0.079 (sd 0.006) | −0.023 | 0.46 |
| 0.10 | 0.120 (sd 0.062) | 0.052 (sd 0.014) | −0.067 | 0.087 |

At 1%: full fine-tuning is 96.1% confident at 71.7% accuracy (badly overconfident); LoRA is
83.4% confident at the same accuracy. Paired ECE gap at 1%: 95% CI [−0.140, −0.114], 5 of 5
seeds agree, Cohen's dz = −12.1. Post-hoc temperature scaling shrinks but does not erase the
gap (1%: −0.128 → −0.040, CI still excludes zero).

**Convergence at 5% — non-overlapping ranges:**

| Fraction | Full FT best epochs | LoRA best epochs | Mean gap | Seeds |
|---|---|---|---|---|
| 0.01 | 5, 5, 7, 8, 8 (mean 6.6) | 7, 7, 8, 8, 8 (mean 7.6) | +1.0 | 3 later, 2 tied |
| 0.05 | 1, 2, 2, 3, 3 | 6, 6, 6, 6, 7 | **+4.0** | **5 of 5, p = 0.0002** |

Full fine-tuning also degrades after its 5% peak, losing 0.0103 ± 0.0056 macro-F1 by
epoch 8, while LoRA barely moves — the reason for the fixed 8-epoch, no-early-stopping
design.

**Same accuracy, different model:** despite equal aggregate accuracy, the two methods agree
on only 84% (1% data) to 87% (5% data) of individual test predictions. At 1%, full
fine-tuning is right where LoRA is wrong on 1,737 items and vice versa on 1,734 — errors
that cancel out to the same score from two measurably different learned functions.

Full details, per-seed numbers, and caveats are in
[`docs/THESIS_MASTER_CONTEXT_v2.md`](docs/THESIS_MASTER_CONTEXT_v2.md).

---

## What NOT to claim from these runs

Every one of these is refuted directly by the data:

- ~~Any test-set claim at 25%, 50% or 100% of the data~~ — nothing at those fractions
  exists under the current protocol; only 1%, 5%, and 10% do.
- ~~"LoRA trains faster"~~ as a general claim — true on a T4 (1.7×), false on an A100
  (~6% slower wall-clock, 1.7–2.7× slower to the checkpoint you keep).
- ~~"LoRA is more stable across seeds"~~ — the standard-deviation direction flips by metric
  and fraction.
- ~~Pooled McNemar~~ — invalid, reuses the same test items five times.
- ~~Anything phrased "as data scales" or "as the training set grows"~~ — three x-values is
  a trio of points, not a scaling curve (the calibration advantage is explicitly
  non-monotone across them).
- ~~The calibration advantage as a general property of LoRA~~ — it's traced to full
  fine-tuning's checkpoint-selection instability under this protocol, not to low-rank
  adaptation itself.

---

## Data-integrity guarantees on the 30 runs

| Check | Result |
|---|---|
| Runs delivered | 30 (fractions 0.01/0.05/0.10, seeds 7/13/21/33/42, both methods) |
| Settings identical across all 30 | fp16, batch 16, ceiling 8, 1 eval/epoch, no patience |
| Early stopping | never fired |
| Evaluation coverage | `evals_ran == evals_possible` in all 30 |
| Checkpoint reload fidelity | `max abs(dev_reload_drift) = 0.000000` across all 30 |
| (fraction, seed) pairs split across GPUs | 0 — every pair shares one GPU |
| Reproducibility check | 6 configurations re-executed identically; up to 0.0068 macro-F1 drift, used as the noise floor above |

---

## Citing

If you use this code or results, please cite the thesis (BibTeX to follow after submission).

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

- CSE-BUET NLP for the [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) model and
  the [XNLI-BN](https://huggingface.co/datasets/csebuetnlp/xnli_bn) dataset.
- Hugging Face `transformers` and `peft` for the training and LoRA implementations.
