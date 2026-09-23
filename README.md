# Bangla NLI — LoRA vs Full Fine-tuning

**MSc thesis code and results.** A matched-budget comparison of LoRA (Low-Rank
Adaptation) against full fine-tuning of [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert)
on the [XNLI-BN](https://huggingface.co/datasets/csebuetnlp/xnli_bn) 3-class natural language
inference task, run at 1% and 5% of the training data with five random seeds per cell (20 runs).

> **Author:** Arman Mahmud · **Base model:** `csebuetnlp/banglabert` · **Task:** 3-class NLI
> (entailment / neutral / contradiction) · **Dataset:** `csebuetnlp/xnli_bn`

---

## Headline result

At both training-set sizes tested, **LoRA and full fine-tuning are statistically equivalent**
on held-out test macro-F1, within **±0.0047 at 1%** and **±0.0096 at 5%** — bounds tighter
than the ±0.0126 binomial noise of a single evaluation on 4,895 test items.

LoRA reaches this parity while training **0.7962% of the parameters** (887,811 of
111,507,462), shipping a **3.57 MB adapter** against a **442.51 MB checkpoint**, reducing
peak GPU memory by **~25%**, and raising training throughput by **1.5–1.9×**.

Two additional findings from the same runs:

- **At 1% data, full fine-tuning is severely overconfident** (ECE 0.244 vs 0.117 for LoRA,
  all 5 seeds in agreement, 95% CI on the gap [−0.140, −0.114]). The effect disappears by 5%.
- **The two methods converge on completely different schedules at 5%**: full fine-tuning
  peaks by epoch 3 and then decays; LoRA peaks at epoch 6 or later — no overlap across seeds.

See [`docs/THESIS_MASTER_CONTEXT.md`](docs/THESIS_MASTER_CONTEXT.md) for the full write-up
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
│   └── THESIS_MASTER_CONTEXT.md                ← full write-up + every number
└── step10_artifacts/                          ← the 20-run evidence base
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
`step9b_equal_budget.py`) were not part of producing the delivered 20 runs and are not
included here.

---

## Reproducing a run

1. Open `notebooks/Step10_retrain_ColabA100.ipynb` in Google Colab on an A100 (bf16) — or
   a T4 (fp16) with the precision cell adjusted. The notebook mounts Drive, installs
   dependencies, and requires **a runtime restart** after the deps cell.
2. Set `PROJECT_DIR` (default `/content/drive/MyDrive/thesis_bangla_nli`). Outputs land in
   `runs/<id>.json`, `curves_v2/<id>.json`, and `preds/<id>_test.npz`.
3. Run all cells. One `(method, fraction, seed)` cell produces one manifest.

Dataset caveats: run counts and wall-clock times listed here are for the 5060 + T4 grid.
Wall-clock and peak-memory columns **cannot be pooled across different GPUs** — the RTX 5060
is ~1.7× faster than the T4 and reports slightly higher peak memory.

---

## Results snapshot (20 runs, 5 seeds per cell)

**Accuracy — bounded null:**

| Fraction | Full FT (test macro-F1) | LoRA (test macro-F1) | Gap (LoRA − full) | 95% CI |
|---|---|---|---|---|
| 0.01 (3,814 examples) | 0.7186 (sd 0.0044) | 0.7182 (sd 0.0079) | −0.0004 | [−0.0060, +0.0053] |
| 0.05 (19,072 examples) | 0.7687 (sd 0.0071) | 0.7697 (sd 0.0041) | +0.0009 | [−0.0104, +0.0123] |

**Efficiency (holds regardless of protocol):**

| Metric | Full fine-tuning | LoRA | Ratio |
|---|---|---|---|
| Trainable parameters | 110,619,651 | 887,811 | **0.7962%** → 125× fewer |
| Artefact on disk | 442.51 MB | 3.57 MB | **124× smaller** |
| Peak GPU memory (T4 median) | 2,420 MB | 1,804 MB | **−25%** |
| Training throughput (samples/s) | 73 / 119 | 140 / 180 | 1.9× at 1%, 1.5× at 5% |

**Efficiency (time-to-best-model — inverts at 5%):**

| Fraction | Full 8-epoch cost | Cost to best model | Verdict |
|---|---|---|---|
| 0.01 | full 7.0 min, LoRA 3.6 min (LoRA −48%) | full 5.2 min (ep 7), LoRA 3.6 min (ep 8) | LoRA −31%, still wins |
| 0.05 | full 21.5 min, LoRA 14.1 min (LoRA −34%) | full **5.4 min** (ep 2), LoRA **11.1 min** (ep 6) | **LoRA +106% — 2× slower** |

**Calibration at 1% — the strongest single finding:**

| Method | Mean confidence | Actual accuracy | ECE |
|---|---|---|---|
| Full fine-tuning | 96.1% | 71.7% | **0.244** |
| LoRA | 83.4% | 71.7% | **0.117** |

Paired ECE gap (LoRA − full) at 1%: **−0.127**, 95% CI [−0.140, −0.114], 5 of 5 seeds
agree, Cohen's dz = −12.1. At 5% the effect is gone (p = 0.45).

**Convergence at 5% — non-overlapping ranges:**

| Fraction | Full FT best epochs | LoRA best epochs | Mean gap | Seeds |
|---|---|---|---|---|
| 0.01 | 5, 5, 7, 8, 8 (mean 6.6) | 7, 7, 8, 8, 8 (mean 7.6) | +1.0 | 3 later, 2 tied |
| 0.05 | 1, 2, 2, 3, 3 | 6, 6, 6, 6, 7 | **+4.0** | **5 of 5, p = 0.0002** |

Full details, per-seed numbers, prediction-agreement (§5.7), validation-vs-test comparison
(§5.8), seed-spread caveats (§5.9), and the abandoned first-pass numbers (§7) are all in
[`docs/THESIS_MASTER_CONTEXT.md`](docs/THESIS_MASTER_CONTEXT.md).

---

## What NOT to claim from these runs

Every one of these is refuted directly by the data:

- ~~Any test-set claim at 10%, 25%, 50% or 100% of the data~~ — nothing at those fractions
  exists under the current protocol.
- ~~"LoRA is 34–48% faster"~~ — inverts at 5% on time-to-best-model.
- ~~"LoRA is more stable across seeds"~~ — the standard-deviation direction flips.
- ~~Pooled McNemar~~ — invalid, reuses the same test items five times.
- ~~Anything phrased "as data scales" or "as the training set grows"~~ — two x-values is not
  a curve.

---

## Data-integrity guarantees on the 20 runs

| Check | Result |
|---|---|
| Runs delivered | 20 (fractions 0.01/0.05, seeds 7/13/21/33/42, both methods) |
| Settings identical across all 20 | fp16, batch 16, ceiling 8, 1 eval/epoch, no patience |
| Early stopping | never fired |
| Evaluation coverage | `evals_ran == evals_possible` in all 20 |
| Checkpoint reload fidelity | `max abs(dev_reload_drift) = 0.000000` across all 20 |
| (fraction, seed) pairs split across GPUs | 0 |

---

## Citing

If you use this code or results, please cite the thesis (BibTeX to follow after submission).

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

- CSE-BUET NLP for the [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) model and
  the [XNLI-BN](https://huggingface.co/datasets/csebuetnlp/xnli_bn) dataset.
- Hugging Face `transformers` and `peft` for the training and LoRA implementations.
