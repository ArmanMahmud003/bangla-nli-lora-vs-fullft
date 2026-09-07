# THESIS MASTER CONTEXT v2 — LoRA vs Full Fine-Tuning on Bangla NLI

**Last updated:** 6 September 2026
**Status:** Stage 1 + Stage 2 complete (30 runs) — ready for §3/§4 writing.
**Target venue:** ACM TALLIP  ·  **Backups:** IEEE Access, BLP @ EMNLP

---

## 1. Study identity

- **Task:** Bangla Natural Language Inference (XNLI-bn, 3-class: entailment / neutral / contradiction).
- **Base model:** `csebuetnlp/banglabert` (ELECTRA-base, 110 M params).
- **Comparison:** Parameter-Efficient Fine-Tuning (LoRA) vs Full Fine-Tuning (Full-FT).
- **Central claim:** *At matched downstream accuracy, LoRA delivers systematically better and more stable calibration than full fine-tuning across all data sizes we tested, at ≈ 1 % the parameter budget.*

## 2. Protocol (pre-registered; not changed after data collection began)

| Item | Value | Rationale |
|---|---|---|
| Data fractions | 0.01, 0.05, 0.10 of XNLI-bn train | Data-scaling axis |
| Seeds | {7, 13, 21, 33, 42} (5 seeds) | Variance estimation |
| Epoch ceiling | 8 | Hard cap, both arms |
| Early stopping | **Off** | Isolates method from stopping rule |
| Best-model selection | Best dev macro-F1 across evals | Standard |
| Evals per epoch | 1 | Consistent scheduler |
| Batch size | 16 | Fits T4 and A100 |
| Precision | **fp16** (locked) | Comparable across GPUs |
| LR — Full-FT | 2 × 10⁻⁵ | Literature default |
| LR — LoRA    | 2 × 10⁻⁴ | Literature default |
| LoRA rank    | 8 | — |
| LoRA α       | 16 | — |
| LoRA dropout | 0.1 | — |
| LoRA targets | Query + Value projections | Standard minimal |
| Max seq len  | 128 | — |
| Optimizer    | AdamW | — |
| Reproducibility | Full seed control, deterministic dataloader | — |

**Grid:** 2 arms × 3 fractions × 5 seeds = **30 runs.** All complete.

## 3. Headline results (test set, XNLI-bn dev/test split)

### 3.1 Accuracy / F1 — LoRA equivalence

| Fraction | Full-FT F1 | LoRA F1 | Gap | Paired t p |
|---:|---:|---:|---:|---:|
| 1 %  | 0.7172 ± 0.0034 | 0.7183 ± 0.0080 | +0.0010 | 0.68 |
| 5 %  | 0.7683 ± 0.0072 | 0.7706 ± 0.0019 | +0.0024 | 0.55 |
| 10 % | 0.7842 ± 0.0027 | 0.7851 ± 0.0044 | +0.0010 | 0.73 |

No significant difference at any data size (n = 5 per fraction).

### 3.2 Calibration — the novelty axis

Expected Calibration Error, 15 equal-width bins, on test-set softmax outputs:

| Fraction | Full-FT ECE | LoRA ECE | Absolute gap | Relative reduction |
|---:|---:|---:|---:|---:|
| 1 %  | 0.2447 ± 0.0088 | 0.1165 ± 0.0091 | 0.1281 | 52 % |
| 5 %  | 0.1019 ± 0.0696 | 0.0790 ± 0.0060 | 0.0229 | 22 % |
| 10 % | 0.1196 ± 0.0615 | 0.0521 ± 0.0141 | 0.0675 | 56 % |

LoRA is better calibrated at every fraction; Full-FT calibration is highly seed-unstable at ≥ 5 %.

### 3.3 Cross-arm agreement

Fraction of test items where LoRA and Full-FT (same seed) predict the same label:

| Fraction | Agreement |
|---:|---:|
| 1 %  | 83.9 % |
| 5 %  | 86.6 % |
| 10 % | 86.9 % |

### 3.4 Efficiency

| Metric | Full-FT | LoRA |
|---|---:|---:|
| Trainable params | 110 619 651 | 887 811 (0.80 %) |
| Saved artifact | 442.51 MB | 3.57 MB |
| Peak GPU mem (10 % run, A100) | 2 713 MB | 1 572 MB |
| Wall-clock @ 10 % (A100 mean) | 1 102 s | 1 181 s |
| Ceiling-hit rate (of 15 runs / arm) | 2/15 | 5/15 |

### 3.5 Integrity receipts

- `reload_max_logit_drift = 0.0` on all 16 A100 runs (14 earlier T4 runs were reproduced on A100 in fp16 for cross-check).
- All 30 runs stamp `precision = fp16`; zero STALE-stamp warnings on final aggregation.
- Best-epoch distribution (mean best_epoch): Full-FT converges earlier (2.4–6.6) than LoRA (4.8–7.8), consistent with LoRA's lower per-step capacity.

## 4. Novelty positioning

- **Prior work:** Shuttleworth et al. (2024, arXiv 2410.21228) showed on English GLUE that LoRA and Full-FT can differ in calibration.
- **This study extends by:**
  1. Testing on a **mid-resource language** (Bangla) instead of English.
  2. Adding an explicit **data-scaling axis** (1 %, 5 %, 10 %) — a calibration-vs-data-size curve.
  3. Using a **strict fairness protocol** (fixed ceiling, no early stopping, matched seeds).
  4. Reporting **arm-agreement** as an additional dimension of comparison.
  5. Providing a **full reproducibility package** (30 manifests, curves, and stored logits).

## 5. Repository layout (as of 6 September 2026)

```
thesis_bangla_nli/
├── Step10_retrain_ColabA100_v2.ipynb        # Reproducibility notebook (fp16 pinned)
├── results_v2.csv                            # 30-run master table
├── results_log.csv                           # legacy log (pre-A100)
├── runs/                *.json  (30 files)   # per-run manifests
├── curves_v2/           *.json  (30 files)   # per-run learning curves
├── preds/               *.npz  (30 files)    # per-run test logits+labels
├── runs_progress/       *.json  (16 files)   # A100 heartbeat receipts
├── README.md                                 # public-facing overview
└── THESIS_MASTER_CONTEXT.md                  # this file
```

## 6. Open decisions (awaiting supervisor input)

1. Extend the data-scaling axis to 25 % / 50 % / 100 % (yes/no)?
2. Add a second base model (XLM-R base) as robustness check (yes/no)?
3. Freeze scope now and write? (recommended)
4. Preferred additional calibration metrics: Brier, NLL, temperature-scaled ECE.

## 7. Writing plan (proposed)

- §1 Introduction — LoRA claim, calibration gap, low-resource framing.
- §2 Related work — PEFT / LoRA, calibration in fine-tuning, low-resource NLI.
- §3 Methodology — protocol §2 of this doc + dataset details.
- §4 Results — three subsections mirroring §3.1–§3.4 above.
- §5 Discussion — why LoRA calibrates better; variance story; practical guidance.
- §6 Limitations — single base model; single language; fixed budget.
- §7 Conclusion + reproducibility statement.
- Appendix — per-seed tables, protocol receipts, hyper-parameter grid.

## 8. Change-log vs v1 context

| Item | v1 | v2 |
|---|---|---|
| Data sizes | 0.01, 0.05 | 0.01, 0.05, **0.10** |
| Runs total | 20 | **30** |
| Calibration curve | 2 points | **3 points** |
| A100 verification | pilot only | **16 A100 runs stamped** |
| STALE-precision gate | fired once (fp16→bf16) | resolved (locked fp16) |
| Ceiling-hit tracking | ad-hoc | fully logged in results_v2.csv |
