# Bangla NLI — LoRA vs full fine-tuning: complete project context

**Compiled 3 September 2026.** Every number below was recomputed from
`thesis_runs_5060_20260827_1404.zip` and machine-checked by `factcheck_md.py` (326 assertions,
zero failures). Where a number comes from the abandoned first pass rather than the real
experiment, it says so.

---

## 1. The whole thing in one page

You are asking whether a cheap fine-tuning method (LoRA) can replace expensive full
fine-tuning on a Bangla natural-language-inference task. You have run that comparison
properly at two training-set sizes — 1% and 5% of the data — with five random seeds each,
twenty runs in total, scored on a held-out test set that was touched exactly once per run.

**The answer you have earned:** at these two data sizes there is no measurable accuracy
difference between the two methods, and the equivalence is tight enough to be quotable
rather than merely "not significant". LoRA gets there while training 0.80% of the
parameters and shipping a 3.57 MB file instead of a 442.51 MB one.

**Two things you have not yet written up, and both are more interesting than the null.**
At 1% data, full fine-tuning becomes wildly overconfident — 96.1% confident while being
71.7% accurate — where LoRA is 83.4% confident at the same 71.7% accuracy. And at 5% the
two methods converge on completely different schedules: full fine-tuning peaks by epoch 3
and then decays, LoRA peaks at epoch 6 or later, with no overlap across seeds at all.

**One thing you must fix before showing anyone.** Your wall-clock speed advantage reverses
at 5% once you measure time to the model you actually keep instead of time to burn the
whole 8-epoch budget. Details in §5.4. This is the single most likely thing to sink you in
a viva or a review.

**Thesis: defensible now**, with corrected framing. Your experimental hygiene is genuinely
above the usual MSc bar. **Paper: not yet** — two data sizes is not a scaling study, and
that is the gap to close with the compute you have.

---

## 2. The question

Full fine-tuning updates all 110 million weights of a pretrained language model. LoRA
(Low-Rank Adaptation) freezes them and inserts small trainable matrices instead, so you
train under a million weights and ship a tiny adapter file. The literature broadly claims
LoRA "matches" full fine-tuning, but that claim is established almost entirely on English
benchmarks with abundant data.

Your contribution question: **does that equivalence survive in a low-resource language,
under a genuinely matched training budget, and does it depend on how much labelled data you
have?** The last clause is what makes it a thesis rather than a replication.

---

## 3. The experiment, exactly as it runs

| Component | Setting |
|---|---|
| Base model | `csebuetnlp/banglabert` (110,619,651 parameters) |
| Task | 3-class natural language inference (entailment / neutral / contradiction) |
| Dataset | `csebuetnlp/xnli_bn`, parquet revision |
| Splits | 381,449 train / 2,419 validation / 4,895 test |
| Test label balance | 1,630 / 1,631 / 1,634 — effectively balanced, so macro-F1 ≈ accuracy |
| Preprocessing | `normalizer.normalize` on both sentences, max_length 128, pad to max |
| Full fine-tuning | all parameters, learning rate 2e-5 |
| LoRA | r=8, alpha=16, dropout 0.1, adapters on `query` and `value`, learning rate 2e-4 |
| Trainable under LoRA | 887,811 of 111,507,462 = **0.7962%** (includes the 3-class head, trained in both methods) |
| Precision | fp16 everywhere, all 20 runs |
| Batch size | 16 train / 64 eval |
| Epoch budget | **fixed 8 epochs for both methods, no early stopping** |
| Evaluation | once per epoch on validation; best epoch chosen afterwards on validation macro-F1 |
| Test scoring | once per run, immediately after the dev-best checkpoint is reloaded |
| Saved per run | manifest `runs/<id>.json`, per-epoch curve `curves_v2/<id>.json`, per-example test logits `preds/<id>_test.npz` |
| Data subsets | `train.shuffle(seed=seed).select(range(n))` — depends only on the seed, so both methods see identical data and subsets nest across fractions |

Two design choices are worth defending out loud because reviewers will ask.

**Why a fixed 8 epochs and no early stopping.** Early stopping with shared patience is not
a shared budget. In your own pilot, patience 2 cut full fine-tuning off at epoch 4 with its
best at epoch 2, while LoRA ran to convergence — because full fine-tuning's validation curve
dips more between peaks. Removing early stopping recovered +0.0161 macro-F1 for full
fine-tuning alone. The fixed ceiling is what makes the comparison honest.

**Why select on macro-F1 and not validation loss.** At 5%, full fine-tuning's validation
loss climbs from 0.77 to 1.54 while its macro-F1 still peaks mid-run. Selecting on loss
would have thrown away its best model.

---

## 4. How the project got here — three passes

### 4.1 First pass (to 22 August): 24 runs, unusable as evidence

Fractions 0.01/0.05/0.10/0.25/0.50/1.00 × seeds {42, 7}. Seven defects, all confirmed
against the files rather than assumed:

1. **No test set.** `eval_dataset` was the 2,419-example validation split; the 4,895-example
   test split was never touched. The reported metric was the best validation score over
   epochs, on the same set used for selection — so every number was selection-inflated.
2. **The inflation was method-asymmetric.** Mean inflation +0.0035 for LoRA vs +0.0020 for
   full fine-tuning. At fraction 0.10 it was +0.0103 vs +0.0000, and the reported LoRA "win"
   there was +0.0110 — i.e. the entire margin was selection noise. On final-epoch values the
   0.10 gap collapses to +0.0008.
3. **Unequal budgets.** Epoch ceilings were LoRA 8/4 versus full fine-tuning 3/2. Nine of
   twelve full-FT runs ended at the ceiling still improving (last-epoch gain up to +0.0197).
4. **Early stopping fired for 4 of 12 LoRA runs and 0 of 12 full-FT runs**, at exactly the
   fractions where LoRA "won". So the low-data comparison was converged LoRA against
   under-trained full fine-tuning.
5. **A reproducibility failure.** `lora_frac0.05_seed7` had no curve and its numbers changed
   between executions at the same seed, traced to `resume_from_checkpoint` picking up stale
   checkpoints.
6. **Only two seeds.** Binomial noise on 2,419 examples is ±1.65 accuracy points, so any gap
   under ~2 points was inside noise. Only the full-FT advantage at ≥0.5 data survived.
7. **`save_total_limit=2` had evicted dev-best checkpoints**, so most runs could not be
   re-scored on test at all — only 3 of 24 checkpoints survived, all LoRA.

The logging code itself was sound: all 23 rows that had curves matched their curves exactly.

### 4.2 The rebuild (23–29 August)

`run_experiment` was replaced by `step10_train_with_test.py`, which fixes every defect above
and adds the instrumentation the first pass lacked: identical 8-epoch ceiling for both
methods, no early stopping, test scored inside each run right after the dev-best checkpoint
is restored, per-example test logits saved, and efficiency logged (trainable parameters,
wall-clock, peak GPU memory, artefact size, library versions, GPU name). One manifest per
run is the source of truth; the CSV is derived from manifests, so a wiped CSV costs nothing.

Guards added, each because something had actually gone wrong: a `CHANCE_F1 = 0.40` floor that
refuses to write a manifest for a collapsed run; `settings_now()` / `stale_runs()`, which
flag any finished run whose manifest disagrees with the current precision, ceiling, batch
size, learning rate or eval cadence; `report_split_risk()`, which warns *before* training
when a run would complete a (fraction, seed) pair whose other half ran on a different GPU;
and an epoch check in the Step 9 re-scoring code, because a surviving checkpoint can name
*itself* "best" once the real best has been evicted.

### 4.3 Second pass (delivered 27 August): the 20 runs you own

Fractions 0.01 and 0.05 × seeds {42, 7, 13, 21, 33} × both methods. 3.44 GPU-hours total.
Sixteen runs on a Colab Tesla T4, four on a local RTX 5060 (fraction 0.05, seeds 21 and 33).
**Fraction 0.10 is not in the archive** despite an earlier belief that it had finished.

---

## 5. Results — every number

### 5.1 Experimental hygiene (your strongest asset, and currently unwritten)

| Check | Result |
|---|---|
| Runs delivered | 20 — fractions 0.01/0.05, seeds 7/13/21/33/42, both methods |
| Training examples | 3,814 (1%) and 19,072 (5%) |
| Settings identical across all 20 | fp16, batch 16, ceiling 8, 1 eval/epoch, `patience_epochs` null |
| Early stopping | never fired — `early_stopped` False in all 20 |
| Evaluation coverage | `evals_ran == evals_possible` in all 20 |
| Checkpoint reload fidelity | `max abs(dev_reload_drift)` = **exactly 0.000000** across all 20 |
| (fraction, seed) pairs split across GPUs | 0 |
| Library drift *within* a pair | 0 (T4: torch 2.11.0+cu128 / transformers 5.15.0; 5060: torch 2.13.0+cu132 / transformers 5.16.1; peft 0.20.0 on both) |

That third-from-last row is the one to put in the thesis. A bit-exact checkpoint reload on
every run means your reported test scores came from precisely the model your validation
metric selected — not an approximation of it. Most student work cannot demonstrate that.

### 5.2 The raw grid — all 20 runs

| Method | Frac | Seed | GPU | Best epoch | Dev F1 | Test acc | Test F1 | Train s | Peak MB | Artefact MB | Samples/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_ft | 0.01 | 7 | T4 | 8 | 0.7183 | 0.7138 | 0.7143 | 411.7 | 2414.6 | 442.51 | 74.2 |
| full_ft | 0.01 | 13 | T4 | 5 | 0.7000 | 0.7156 | 0.7173 | 471.6 | 2413.4 | 442.51 | 65.1 |
| full_ft | 0.01 | 21 | T4 | 8 | 0.7150 | 0.7140 | 0.7148 | 560.8 | 2421.3 | 442.51 | 54.5 |
| full_ft | 0.01 | 33 | T4 | 7 | 0.7284 | 0.7213 | 0.7229 | 360.0 | 2421.3 | 442.51 | 84.9 |
| full_ft | 0.01 | 42 | T4 | 5 | 0.7164 | 0.7218 | 0.7235 | 419.3 | 2424.7 | 442.51 | 73.5 |
| lora | 0.01 | 7 | T4 | 8 | 0.7018 | 0.7107 | 0.7116 | 218.2 | 1806.0 | 3.57 | 140.1 |
| lora | 0.01 | 13 | T4 | 7 | 0.7039 | 0.7199 | 0.7207 | 218.4 | 1805.8 | 3.57 | 140.0 |
| lora | 0.01 | 21 | T4 | 8 | 0.7156 | 0.7079 | 0.7086 | 219.1 | 1803.5 | 3.57 | 139.6 |
| lora | 0.01 | 33 | T4 | 8 | 0.7242 | 0.7269 | 0.7279 | 217.7 | 1804.3 | 3.57 | 140.4 |
| lora | 0.01 | 42 | T4 | 7 | 0.7193 | 0.7218 | 0.7222 | 217.6 | 1800.3 | 3.57 | 140.5 |
| full_ft | 0.05 | 7 | T4 | 2 | 0.7603 | 0.7749 | 0.7759 | 1287.4 | 2422.9 | 442.51 | 118.8 |
| full_ft | 0.05 | 13 | T4 | 3 | 0.7713 | 0.7642 | 0.7649 | 1396.1 | 2414.5 | 442.51 | 109.5 |
| full_ft | 0.05 | 21 | 5060 | 2 | 0.7682 | 0.7704 | 0.7707 | 738.5 | 2462.2 | 442.51 | 207.0 |
| full_ft | 0.05 | 33 | 5060 | 3 | 0.7621 | 0.7726 | 0.7738 | 719.1 | 2468.0 | 442.51 | 212.5 |
| full_ft | 0.05 | 42 | T4 | 1 | 0.7656 | 0.7589 | 0.7583 | 1275.7 | 2419.5 | 442.51 | 119.8 |
| lora | 0.05 | 7 | T4 | 6 | 0.7633 | 0.7712 | 0.7721 | 844.2 | 1801.1 | 3.57 | 180.8 |
| lora | 0.05 | 13 | T4 | 7 | 0.7755 | 0.7700 | 0.7707 | 847.4 | 1803.2 | 3.57 | 180.1 |
| lora | 0.05 | 21 | 5060 | 6 | 0.7705 | 0.7618 | 0.7625 | 531.7 | 1858.8 | 3.57 | 287.1 |
| lora | 0.05 | 33 | 5060 | 6 | 0.7744 | 0.7700 | 0.7703 | 529.6 | 1862.2 | 3.57 | 288.3 |
| lora | 0.05 | 42 | T4 | 6 | 0.7729 | 0.7716 | 0.7728 | 884.3 | 1808.5 | 3.57 | 172.6 |

Note the wall-clock and peak-memory columns cannot be pooled across the two GPUs — the 5060
is roughly 1.7× faster and reports slightly higher peak memory. Every timing figure quoted
in this document is **T4 only**, which is the sixteen-run subset.

### 5.3 Accuracy — a bounded null, not "no significant difference"

Mean test macro-F1 by method:

| Fraction | Full fine-tuning | LoRA | Gap (LoRA − full) |
|---|---|---|---|
| 0.01 (3,814 examples) | 0.7186 (sd 0.0044) | 0.7182 (sd 0.0079) | **−0.0004** |
| 0.05 (19,072 examples) | 0.7687 (sd 0.0071) | 0.7697 (sd 0.0041) | **+0.0009** |

Paired seed-level statistics on that gap:

| Fraction | Mean gap | 95% CI | p | Equivalent within | Detectable at 80% power |
|---|---|---|---|---|---|
| 0.01 | −0.0004 | [−0.0060, +0.0053] | 0.87 | ±0.0047 (0.47 points) | 0.0076 |
| 0.05 | +0.0009 | [−0.0104, +0.0123] | 0.83 | ±0.0096 (0.96 points) | 0.0152 |

**The framing that makes this publishable.** A single run's accuracy on 4,895 test items
carries binomial noise of ±1.26 points at 72% accuracy. Both of your equivalence bounds are
*tighter than that*. So the sentence is not "we found no significant difference" — it is:
*the two methods are indistinguishable to a precision finer than the test set can resolve
for any single run.* That costs nothing to say and is a real, defensible claim.

Per-seed test gaps, for the appendix — 1%: −0.0014, −0.0027, +0.0034, −0.0061, +0.0050
(seeds 42, 7, 13, 21, 33). 5%: +0.0145, −0.0038, +0.0057, −0.0082, −0.0036.

### 5.4 Efficiency — three claims hold, one inverts

**These hold, and do not depend on your protocol at all:**

| Metric | Full fine-tuning | LoRA | Ratio |
|---|---|---|---|
| Trainable parameters | 110,619,651 | 887,811 | 0.7962% → 125× fewer |
| Saved artefact on disk | 442.51 MB | 3.57 MB | **124× smaller** |
| Peak GPU memory (T4 median) | 2,420 MB | 1,804 MB | **−25%** |
| Training throughput | 73 / 119 samples/s | 140 / 180 samples/s | 1.9× at 1%, 1.5× at 5% |

**This one reverses.** `train_runtime_s` measures all 8 epochs. But the model you ship is
the dev-best epoch, and the two methods peak at very different times — so the fair cost is
`runtime × best_epoch / 8`:

| Fraction | Cost of the full 8-epoch budget | Cost to reach the model you keep | Verdict |
|---|---|---|---|
| 0.01 | full FT 7.0 min, LoRA 3.6 min (LoRA −48%) | full FT 5.2 min (ep 7), LoRA 3.6 min (ep 8) | LoRA −31%, still wins |
| 0.05 | full FT 21.5 min, LoRA 14.1 min (LoRA −34%) | full FT **5.4 min** (ep 2), LoRA **11.1 min** (ep 6) | **LoRA +106% — 2× slower** |

Full fine-tuning at 5% reaches its best model at epoch 2 and then gets *worse*. LoRA needs
epoch 6. So "LoRA trains 34% faster" is an artefact of forcing both methods through eight
epochs. Your manifests contain `best_epoch`, so any referee can do this arithmetic in about
a minute. Report both columns yourself and the criticism becomes a strength.

### 5.5 Calibration — your strongest finding, and it only exists at 1%

"Calibration" means: when a model says it is 90% sure, is it right 90% of the time?
Expected calibration error (ECE) measures the average size of that mismatch; lower is
better. You can compute all of this for free from the test logits you already saved.

| Fraction | Method | Mean confidence | Actual accuracy | ECE |
|---|---|---|---|---|
| 0.01 | full fine-tuning | **96.1%** | 71.7% | **0.244** |
| 0.01 | LoRA | 83.4% | 71.7% | **0.117** |
| 0.05 | full fine-tuning | 86.5% | 76.8% | 0.099 |
| 0.05 | LoRA | 84.8% | 76.9% | 0.080 |

| Fraction | Paired ECE gap (LoRA − full) | 95% CI | Seeds agreeing | Effect size |
|---|---|---|---|---|
| 0.01 | **−0.1267** | [−0.1397, −0.1138] | **5 of 5** | Cohen's dz = −12.1 |
| 0.05 | −0.019 | [−0.082, +0.044] | 4 of 5 | p = 0.45 — not there |

Supporting evidence at 1%: test negative log-likelihood gap −0.571, 95% CI [−0.722, −0.421].
The result is identical under 10 equal-width bins, 15 equal-width bins, and 15 equal-mass
bins, so it is not a binning artefact.

**The obvious objection, pre-rebutted.** "LoRA looks better calibrated because it trained
less." No — LoRA trained *longer*: mean best epoch 7.6 versus 6.6 at 1%.

**State the scope honestly.** This is a low-data effect. It is gone at 5% (p = 0.45). Do not
write "LoRA is better calibrated than full fine-tuning"; write "at 1% of the data, full
fine-tuning becomes severely overconfident while LoRA does not, and the effect disappears by
5%." The second sentence is both true and more interesting.

### 5.6 Convergence — the cleanest separation in your data

| Fraction | Full FT best epochs | LoRA best epochs | Mean gap | 95% CI | Seeds |
|---|---|---|---|---|---|
| 0.01 | 5, 5, 7, 8, 8 (mean 6.6) | 7, 7, 8, 8, 8 (mean 7.6) | +1.0 | [−0.24, +2.24] | 3 later, 2 tied, 0 earlier; p = 0.089 |
| 0.05 | 1, 2, 2, 3, 3 | 6, 6, 6, 6, 7 | **+4.0** | [+3.12, +4.88] | **5 of 5, p = 0.0002** |

At 5% the two ranges do not overlap at all — every full-FT run peaks by epoch 3, every LoRA
run peaks at 6 or later. And full fine-tuning does not merely plateau, it decays: validation
macro-F1 from peak to epoch 8 falls by 0.0117 for full fine-tuning versus 0.0028 for LoRA
(gap −0.0090, p = 0.010). At 1% the decay is 0.0055 vs 0.0010 (p = 0.074).

This is why a shared early-stopping patience was impossible and had to be removed — and that
makes it a methodological contribution, not just an observation.

**One caveat to state.** At 1%, five of the ten runs (two full FT, three LoRA) peaked at
epoch 8, so the budget is mildly binding there — but symmetrically, and the last-epoch gains
are small (+0.0004 to +0.0098). Nothing hit the ceiling at 5%.

### 5.7 Prediction agreement — same score, different function

Nobody has written this up and it is one sentence away from being interesting.

| Fraction | Items both get right or both wrong | LoRA-only correct (B) | Full-FT-only correct (C) |
|---|---|---|---|
| 0.01 | 83.7% | 1,737 | 1,734 |
| 0.05 | 86.8% | 1,433 | 1,416 |

The two methods disagree on one test item in six, and their errors cancel almost exactly.
Equal scores are not the same model. This connects directly to *"LoRA vs Full Fine-tuning:
An Illusion of Equivalence"* (arXiv 2410.21228) and gives your null result a mechanism
rather than leaving it as an absence.

**On the significance tests.** A pooled McNemar test across the five seeds (p = 0.973 at 1%,
0.764 at 5%) is *not valid* — it reuses the same 4,895 test items five times and understates
the true variance. Run it per seed instead. Per-seed exact McNemar p-values, same seed order as §5.3 (42, 7, 13,
21, 33), are 1.000, 0.596, 0.429, 0.271, 0.342 at 1% and **0.012**, 0.473, 0.258, 0.089,
0.609 at 5%. That one
significant result is 1 of 10 tests, favours LoRA, is contradicted by the other four seeds at
the same fraction, and dies under Holm correction (p = 0.12). Do not quote it as a finding.

### 5.8 Validation versus test — direct proof the first pass flattered LoRA

Same runs, same checkpoints, only the split changes:

| Fraction | Validation gap (LoRA − full) | LoRA ahead on dev | Test gap | LoRA ahead on test |
|---|---|---|---|---|
| 0.01 | −0.0027 | 3 of 5 | −0.0004 | 2 of 5 |
| 0.05 | **+0.0058** | **5 of 5** | **+0.0009** | 2 of 5 |

At 5%, LoRA leads on validation in every single seed and on test in only two. That is a clean
demonstration of why a dev-only protocol — the first pass's protocol — was misleading, and it
justifies the entire rebuild in one table.

### 5.9 Seed spread — do not claim LoRA is more stable

| Fraction | Full FT test F1 sd | LoRA test F1 sd |
|---|---|---|
| 0.01 | 0.0044 | 0.0079 |
| 0.05 | 0.0071 | 0.0041 |

The direction flips between fractions. With five seeds you cannot tell these apart. Any
"LoRA is more stable" sentence is unsupported.

---

## 6. What you may claim, and what you must not

**Claimable, all verified:**

- No measurable accuracy difference at 1% and 5%, equivalent within ±0.0047 and ±0.0096 —
  bounds tighter than single-run binomial noise (±0.0126).
- 0.7962% of parameters trained; 124× smaller artefact; −25% peak GPU memory; 1.5–1.9×
  training throughput.
- At 1% data, full fine-tuning is severely overconfident (ECE 0.244 vs 0.117, 5/5 seeds);
  the effect vanishes at 5%.
- Full fine-tuning converges ~4 epochs earlier at 5% (non-overlapping ranges) and then
  degrades four times faster than LoRA past its peak.
- The methods agree on only 83.7% / 86.8% of test items with symmetric errors.
- Bit-exact checkpoint reload on all 20 runs; no early stopping; complete eval coverage.

**Not claimable, and each of these would be caught:**

- ~~Any test-set claim at 10%, 25%, 50% or 100% of the data~~ — **nothing at 10%, 25%, 50% or
  100% exists under the current protocol.** Every number above 5% anywhere in this document is
  first-pass, validation-only and labelled as such in §7, or a future-work estimate in §10.3.
- ~~"LoRA is 34–48% faster"~~ — inverts at 5% on time-to-best-model. §5.4.
- ~~"LoRA is more stable across seeds"~~ — the standard deviation flips direction. §5.9.
- ~~Pooled McNemar~~ — invalid, reuses the same test items five times. §5.7.
- ~~The seed-42 p = 0.012 result~~ — 1 of 10 tests, dies under Holm.
- ~~Per-class differences~~ — sign counts across seeds are 3/5, 1/5, 3/5. Noise.
- ~~Anything phrased "as data scales" or "as the training set grows"~~ — you have two
  x-values. Two points is not a curve. This is the single biggest barrier to publication.
- ~~Comparing your `full_ft_frac1.0_seed42` dev accuracy of 0.8280 to the BanglaBERT paper's
  reported 82.80 test accuracy~~ — coincidence, different splits, not a reproduction.

---

## 7. First-pass numbers for the larger fractions — context only, not results

These are validation-only, unequal-budget, two-seed numbers from the abandoned first pass.
They are the reason you believe there may be a crossover, and they are worth keeping as
motivation — but they cannot appear in a results section. LoRA minus full fine-tuning,
validation macro-F1, as-run versus re-read from the curves under the smaller budget that full
fine-tuning actually received at each fraction (3 epochs at 1–10%, 2 epochs at 25–100%):

| Fraction | Equal-budget gap | As-run gap |
|---|---|---|
| 0.01 | −0.0095 | +0.0091 |
| 0.05 | −0.0056 | +0.0010 |
| 0.10 | +0.0046 | +0.0110 |
| 0.25 | −0.0095 | −0.0051 |
| 0.50 | −0.0308 | −0.0214 |
| 1.00 | −0.0218 | −0.0134 |

One caveat on the 0.05 row: the per-epoch curve for `lora_frac0.05_seed7` did not survive into
`all_curves.json`, so its equal-budget cell averages one LoRA seed against two full-FT seeds
while its as-run neighbour uses both. Every other cell is a clean two-seed mean.

Under an equal epoch budget the low-data LoRA advantage disappears entirely. The honest
reading is that equal-epoch framing favours full fine-tuning while the original 8/4-vs-3/2
budget favoured LoRA, and the truth is bracketed between them. The full-FT advantage at ≥0.5
data (−0.02 to −0.03) is the one first-pass signal large enough to survive its own noise —
which is precisely why the 25/50/100% runs are the highest-value experiments left.

Caveat on the equal-budget column: capping the curves at the full-FT budget compares full
fine-tuning on a fully annealed 3- or 2-epoch schedule against LoRA at that same epoch of an
8-epoch schedule, whose learning rate has not yet decayed. It is a proxy, mildly unfair to
LoRA. Only real runs with matched ceilings settle it — which is what the second-pass protocol
does.

---

## 8. Literature review — current status

Two A3-landscape summary tables, seven studies each, fourteen columns. `literature_dataset
(1).pdf` is the on-topic one (LoRA/PEFT for Bangla, plus BanglaBERT and Bangla sentiment);
`Bangla Research Literature Dataset - PDF Format.pdf` is mixed relevance (cuisine
recognition, code-mixed word recognition, Bangla medical NER, agricultural RAG).

**Confirmed defects to fix:**

- Study #5 (Ecstasy, BLP-2025) row is shifted one column: preprocessing text sits in Data
  Splits, metrics sit in Preprocessing, results sit in Evaluation Metrics, and Best Results
  contains numbers (96.86% accuracy, AUC ≈ 1.00, 294K trainable params, 8.2 GB) that actually
  belong to a study in the *other* PDF.
- Page 8 clips 72 words past the right page edge (Study #7's last two columns); page 9 is blank.
- Study #4 (CodeAnubad) Workflow cell is written in transliterated Banglish; every other cell
  is English.
- Study #2's key finding claims PEFT beat full fine-tuning, but that paper's own Models column
  shows only LoRA vs QLoRA — there is no full-FT baseline in it.
- In the second PDF, Study #2's Limitations cell is a near-duplicate of its Best Results cell,
  so the real limitations are missing; row 1 has no study number.

**Coverage gaps — these must be filled before submission.** Neither review cites the LoRA
paper (Hu et al. 2021), adapters (Houlsby et al. 2019), BitFit, or XNLI (Conneau et al. 2018).
None of the fourteen studies performs your actual experiment — PEFT versus full fine-tuning
across training-set sizes — and none works on NLI. For the paper you additionally need the
calibration literature: Laplace-LoRA (arXiv 2308.13111), arXiv 2410.06431, the LoRA/full-FT
calibration comparison on GLUE (arXiv 2603.19278), and "An Illusion of Equivalence"
(arXiv 2410.21228).

Also note two easily confused papers, both present in your files: Bhattacharjee et al.'s
BanglaBERT (`csebuetnlp/banglabert` — the model you actually use) and Kowsher et al.'s
Bangla-BERT. Do not cite one for the other.

---

## 9. To finish the thesis

Nothing here needs a GPU. All of it is writing and re-analysis of files you already have.

**Must fix, in this order:**

1. **Rebuild the efficiency section around time-to-best-model.** Report both cost columns
   from §5.4 and state the inversion explicitly. Highest-risk item in the document.
2. **Restate the null as an equivalence bound.** Replace every "no significant difference"
   with the ±0.0047 / ±0.0096 bounds and the comparison to ±1.26 points of single-run noise.
3. **Replace pooled McNemar with per-seed McNemar plus Holm correction**, and drop the
   seed-42 p = 0.012 claim.
4. **Add the two unwritten findings** — the 1% calibration gap (§5.5) and the 5% convergence
   separation (§5.6). Both are free: they come from logits and curves you already saved.
5. **Add a scope paragraph.** State plainly that you have two data sizes, that this is not a
   scaling study, and that the 10/25/50/100% points are future work. Say it before a
   reviewer says it for you.
6. **Add the hygiene paragraph** from §5.1 — bit-exact reload, no early stopping, complete
   eval coverage, no split pairs. This is a genuine differentiator and it is currently invisible.
7. **Add the agreement observation** (§5.7) as one or two sentences in the discussion.
8. **Fix the two literature-table data errors** and add the four missing foundational citations.
9. **Delete or clearly re-label every first-pass number** still in the draft. Anything from
   `results_log (1).csv` is validation-only, unequal-budget, two-seed. If it appears in a
   results table, it will be treated as a result.
10. **Add a limitations section** that names: two data fractions, five seeds, one LoRA rank,
    one learning rate per method, one base model, one task, one language.

**Four sentences you can paste more or less as they stand:**

> At both training-set sizes tested, LoRA and full fine-tuning are statistically equivalent
> on held-out test macro-F1, within ±0.0047 at 1% and ±0.0096 at 5% — bounds tighter than the
> ±0.0126 binomial noise of a single evaluation on 4,895 items.

> LoRA reaches this parity while training 0.7962% of the model's parameters, shipping a
> 3.57 MB adapter against a 442.51 MB checkpoint, reducing peak GPU memory by 25%, and raising
> training throughput by 1.5–1.9×. Wall-clock time to the selected model, however, favours
> full fine-tuning at 5% (5.4 versus 11.1 minutes), because full fine-tuning peaks at epoch 2
> while LoRA peaks at epoch 6.

> At 1% of the data, full fine-tuning is 96.1% confident while being 71.7% accurate, whereas
> LoRA is 83.4% confident at the same accuracy; the expected-calibration-error gap is −0.127
> (95% CI [−0.140, −0.114], all five seeds in agreement) and disappears by 5% of the data.

> Both methods were trained for eight epochs at every data size with no early stopping; the
> epoch with the highest validation macro-F1 was selected and evaluated once on the test split,
> with a measured checkpoint-reload drift of 0.000000 across all twenty runs.

---

## 10. To make it a publishable paper

### 10.1 Where the novelty actually sits

Checked against the current literature, honestly:

- LoRA/full-FT **calibration parity** is already published on GLUE (arXiv 2603.19278).
- **Overconfidence under sparse fine-tuning data** is a crowded area (Laplace-LoRA
  arXiv 2308.13111, arXiv 2410.06431).
- **"Equal score, different function"** is covered by arXiv 2410.21228.

So your calibration result is a *replication with a data-scale twist on a low-resource
language* — not a new phenomenon. Claim it at exactly that level and it is publishable and
citable. Overclaim it and a reviewer who knows the area will reject on novelty alone.

**What is genuinely yours to own:** the data-size dependence. Nobody has shown *when* the
calibration gap appears and disappears, and your two points already suggest it is a low-data
effect that closes by 5%. Turn that into a curve and you have a paper-sized contribution:
**"the LoRA/full-FT calibration gap is a low-data phenomenon and closes as data grows."**

### 10.2 Realistic venue

**As currently framed: reject.** Not because of quality but because two data sizes cannot
support the claim the title would need to make.

**Reframed, with the scaling axis filled in:** the BLP (Bangla Language Processing) workshop
at EMNLP, a low-resource-language workshop such as LoResLM, or a regional journal. Not an
ACL/EMNLP main track — and saying so now saves you a wasted submission cycle.

### 10.3 Experiment roadmap, cheapest and most valuable first

You have weeks of GPU time on the RTX 3070 and the staged runner already implements this grid.

| Stage | What to run | Hours | What it buys |
|---|---|---|---|
| **A — do this first** | Fractions 10%, 25%, 50%, 100% × 5 seeds where affordable (drop to 3 seeds at 50%, 1 at 100% if the clock bites) | 35–45 | Turns two points into a curve. Without this there is no paper. Directly tests whether the calibration gap closes and whether full FT overtakes at scale — which your first pass hints at (−0.02 to −0.03 at ≥50%). |
| **B — do this second** | Learning-rate sweep per method: 3 LRs × 2 methods × 3 seeds at 5% | 5–6 | Kills the single most likely referee objection: "you used 2e-5 and 2e-4 defaults, so you never tuned full fine-tuning." Right now your answer is a scope sentence; after this it is a table. |
| **C** | Extend 1% and 5% from 5 to 10 seeds | 8–12 | Cuts the minimum detectable effect by ~30% (0.0076 → ~0.0054 at 1%), tightening the equivalence bounds you are selling. |
| **D** | LoRA rank sweep, r ∈ {2, 8, 32} at one or two fractions | 9–12 | Stops "LoRA" meaning "one configuration I picked". Also probes whether the calibration gap is a capacity effect. |
| **E — free, no GPU** | Recompute calibration, time-to-best and prediction agreement for every new run from saved logits and curves | 0 | Every finding in §5.5–5.7 extends automatically to the new fractions. |

Stage A alone converts this from "a careful null result at two data sizes" into "a
characterisation of when PEFT equivalence holds in a low-resource language". That is the
difference between a thesis and a paper.

**Order matters.** Run A before B. If the curve shows full fine-tuning pulling ahead at
50–100%, the paper's story changes and B's design should follow it.

### 10.4 Paper skeleton once stage A lands

**Title direction:** *When does parameter-efficient fine-tuning match full fine-tuning? A
data-scale study on Bangla natural language inference.*

**Contributions, in the order a reviewer will weigh them:**

1. A matched-budget PEFT-vs-full-FT comparison across the full data-size range on a
   low-resource language, with per-example predictions released.
2. The calibration gap as a function of training-set size — the novel axis.
3. The convergence-schedule separation, and the methodological consequence that a shared
   early-stopping patience is not a shared budget.
4. Equivalence bounds rather than failed null hypothesis tests, benchmarked against the
   resolving power of the test set.

**Sections:** the null and its bounds → efficiency with both cost definitions →
calibration versus data size (the figure that carries the paper) → convergence and post-peak
decay → prediction agreement → limitations.

**Artefacts to release:** the twenty-plus manifests, per-epoch curves, and per-example test
logits. Very few workshop papers in this space ship per-example predictions; yours already
exist and cost nothing to publish.

---

## 11. Code and file inventory

**Training and analysis (the current, correct pipeline):**

| File | What it does |
|---|---|
| `step10_train_with_test.py` | The real trainer. Matched 8-epoch ceiling, no early stopping, test scored in-run after dev-best reload, per-example logits saved, efficiency logged, one manifest per run. Also holds `import_zip` / `export_zip` / `stale_runs` / `report_split_risk`. |
| `step3_data.py` | Data loading, normalisation and tokenisation, extracted so the notebook and the headless runner can never diverge. |
| `step11_analysis.py` | Consistency checks, test macro-F1 tables, paired per-seed gaps, McNemar and bootstrap CIs, efficiency table, two figures. |
| `run_all_3070.py` + `.bat` | Unattended staged training with a PID lock, per-start log, preflight refusals, one automatic retry, a backup zip after every finished run, and exit codes the batch file branches on. Stages 2–5 = 10% / 25% / 50% / 100%. |
| `check_progress.py` | What is finished or missing per stage, which GPU ran it, split-pair flags. **Run this on the 3070 first.** |
| `merge_zip.py` | Standalone Colab-side merge for a session holding an older notebook. |

**Notebook variants** — four, differing only in hardware settings:
`Step10_retrain_ColabT4.ipynb` (fp16), `Step10_retrain_ColabA100.ipynb` (bf16),
`Step10_retrain_Local5060.ipynb` (fp16, Blackwell needs the CUDA 12.8 wheel),
`Step10_retrain_Local3070.ipynb` (fp16, Ampere, any cu12x wheel works).

**Superseded, keep for provenance only:** `Thesis.ipynb - Colab.pdf`, `results_log (1).csv`,
`all_curves.json`, `step9_test_eval.py`, `step9b_equal_budget.py`, `Step9_test_eval.ipynb`.

**Data and documents:** `thesis_runs_5060_20260827_1404.zip` (the 20 runs — this is the
evidence base), `results_v2_asdelivered.csv`, `significance_1pct_5pct.csv`,
`REVIEW_VERDICT.pdf` (the seven-page peer-review verdict), `SETUP_GUIDE_3070.pdf`,
`SUPERVISOR_BRIEF_2026-08-27.pdf`, `THESIS_FIX_LIST.md`, `THESIS_TODO.pdf`, and the two
literature-review PDFs.

**Transfer rules that must not be relaxed:** never split a (fraction, seed) pair across two
GPUs — accuracy pools across machines but a paired gap does not, and wall-clock and peak
memory never pool. Divide work by seed. `results_v2.csv` is never transferred; it is rebuilt
from manifests, which removes the only file two machines would both write.

---

## 12. Open loose ends

- **Fraction 0.10 does not exist anywhere.** It was believed finished; the archive contains
  nothing at that fraction. Run `check_progress.py` on the 3070 before assuming any stage is
  complete. This is stage A's first job anyway.
- **A salvageable checkpoint.** The interrupted 100%-data run left a scratch checkpoint in
  `D:\thesis_ckpt_scratch` on the *previous* machine. The gated resume can use it if its
  settings stamp matches the current configuration. Do not let anything delete it before that
  is checked — it is the single most expensive run in the grid.
- **Environment gotchas worth remembering.** `peft` raises `ImportError: Found an incompatible
  version of torchao` when loading an adapter, so `pip install torchao --upgrade` is required.
  The `classifier.* MISSING` load report from `csebuetnlp/banglabert` is normal and harmless,
  as is `use_return_dict is deprecated` and the early-stopping metric warning triggered by
  manual `evaluate()` / `predict()` calls.
- **fp16 is verified safe** on this task: the fp16 pilot matched the fp32 pilot to within
  0.0058 macro-F1 and its training-loss curve tracked fp32 to within 0.009 per epoch. The
  `CHANCE_F1 = 0.40` guard catches a collapse if one ever happens.

---

## 13. Reproducing every number in this document

All analysis scripts are pure numpy and pandas — **scipy and sklearn are not installed**, so
the statistics are hand-rolled: the regularised incomplete beta function via a Lentz
continued fraction, Student-t quantiles by bisection, two-sided exact binomial McNemar in log
space, TOST equivalence from the 90% confidence interval, and Monte-Carlo t-quantiles plus a
Monte-Carlo power bisection used as an independent check on the analytic route.

| Script | Purpose |
|---|---|
| `review_evidence.py` | Grid integrity, paired seed-level gaps, agreement, McNemar, efficiency, convergence, binomial yardstick |
| `review_evidence2.py` | Time-to-best-model (found the inversion), per-class F1, ECE, seed spread, ceiling-binding counts |
| `review_evidence3.py` | Confidence intervals on the calibration, NLL and convergence findings |
| `factcheck.py` | Re-derives all 105 published numbers by independent routes and asserts each appears in the verdict PDF |
| `factcheck_md.py` | Does the same for this document — 326 checks, including all 240 cells of the §5.2 grid, every interval in §5.3–5.9, the §7 first-pass table, and a scan proving no claim rests on a run that was never made |
| `dump_tables.py` | Regenerates the raw grid and every table in §5 |

Both fact-checkers currently pass with zero failures. They found four real errors on the way
there, which is the reason to keep running them: a peak-memory median quoted as 2,421 MB
instead of 2,420, one dev-F1 cell mistyped as its test-F1 neighbour, an effect size rounded to
−12.2 when it is −12.1, and a per-seed p-value list printed in a different seed order from
every other list in the document.

The single source of truth is `thesis_runs_5060_20260827_1404.zip`: `runs/*.json` for
per-run facts, `curves_v2/*.json` for per-epoch validation curves, `preds/*_test.npz` for
per-example test logits and labels. Every finding in §5 can be rebuilt from those three
folders with no GPU.

---

## Bottom line

You have a defensible thesis today provided you correct the efficiency framing, restate the
null as an equivalence bound, and add the two findings sitting unused in your saved logits.
You do not yet have a paper, and the reason is a single missing axis: data scale. Roughly
40 hours of GPU time on runs the code is already written to execute closes that gap, and
your existing analysis pipeline will pick up every new run automatically.

The strongest thing about this work is not any individual result — it is that the protocol is
tight enough for the null result to mean something. Very few student projects can say their
reported test score came from a bit-exact reload of the checkpoint their validation metric
chose. Lead with that.
