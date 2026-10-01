# 11. Gap analysis

This document is a root cause analysis of the gap between the evaluation results of the Lizard model and the results of the paper. A review examined these sources:

- the notes in `docs/`,
- the Lizard implementation,
- the LoLCATs training configs,
- the code for evaluation and for model loading.

The gap looks **real**. The review does not treat it as mainly a problem of the MMLU evaluation.

Sections 1–10, the hypothesis and the debugging sequence come from the review. Sections 11 and 12 come from two later analyses: a guideline to close the gap, and the contribution of each hyperparameter. Section 13 ranks the causes of the "A" collapse after the stage difference experiment, and gives a debugging plan. A paragraph that starts with **Repository check** gives the result of a check against the code or the other documents. These checks occurred during the writing of this document.

## The gap

| Benchmark | Lizard model | Paper, Lizard 1B | Gap |
|---|---:|---:|---:|
| MMLU 5-shot | 23.3 | 29.8 | **−6.5** |
| PIQA 0-shot | 68.0 | 74.8 | **−6.8** |
| ARC-Easy 0-shot | 54.8 | 65.6 | **−10.8** |

Table 9 of the paper gives these values for Llama-3.2-1B and describes Lizard as close to the teacher. In the paper, the teacher gets 31.0 on MMLU, 74.1 on PIQA and 65.4 on ARC-Easy.

The important clue is that **all three tasks show damage**. On MMLU, the model also has a very specific symptom: it almost always selects "A". This suggests a problem in the training or in the model quality first. The MMLU behavior is an additional, MMLU-specific effect.

## Ranking of the possible causes

The benchmark columns show how strongly each cause can affect that benchmark.

| Priority | Possible cause | MMLU | PIQA | ARC-Easy | Assessment |
|---|---|---:|---:|---:|---|
| **1** | The stage 1 recipe is very different from the paper | High | High | High | **Most probable** |
| **2** | The stage 2 recipe and the frozen Lizard parameters are different from the paper | High | High | High | **Very plausible** |
| **3** | The LoRA checkpoint has missing keys, and the loader does not report them | High | High | High | **Check immediately** |
| **4** | The training data, the packing, the tokenizer or the prompts do not match the paper | High | High | High | Plausible |
| **5** | Numerical differences in the gated branch and the gate at full scale | High | High | High | Plausible |
| **6** | Differences in the harness version and in the tokenization of the evaluation | High | Medium | Medium | Real, but probably not the explanation for all of the gap |
| **7** | A formatting or tokenization problem with the MMLU answer letters | **Very high** | – | – | Explains the symptom, but is probably not the root cause |
| **8** | Checkpoint selection, seed, or variance from run to run | Medium | Medium | Medium | Possible |
| **9** | Differences between the Llama implementations of transformers versions | Medium | Medium | Medium | Exclude it |

Related documents: [results](07-results.md), [comparison with the paper](09-paper-comparison.md), [evaluation setup](06-evaluation-setup.md).

## 1. Stage 1: the largest warning sign

The runs do **not use the recipe of the paper**.

| Setting | Paper (Table 13) | Runs of this project |
|---|---|---|
| Stage 1 learning rate | **1e-3** | **1e-2** |
| Stage 2 learning rate | **5e-4** | **1e-4** |
| Schedule | Cosine | ReduceLROnPlateau |
| Warmup | 10%, linear | None |
| AdamW β | (0.9, 0.99) | β2 = **0.999** |
| Gradient clipping | 1.0 | None |
| Global batch | 8 | Same as the paper |
| Sequence length | 2048 tokens | Same as the paper |
| LoRA targets | Wq, Wk, Wv | q, k, v, **o** |
| Lizard parameters in stage 2 | No information | Frozen |

This is not a small deviation for a reproduction.

The most important difference: **stage 1 starts with new feature maps and a new gate. Its learning rate is 10× the final learning rate of the paper, with no warmup and no clipping.**

**Repository check:** The review describes the feature maps and the gate as randomly initialized. In `src/model/linear_attention/lizard_attention.py`, the feature maps start with random values (normal distribution, standard deviation 0.02). The gate starts at 0, so γ = 0.5 for every token ([document 1](01-lizard-in-lolcats.md)).

The paper says that stage 1 trains the feature maps φq, φk and the gate matrix Wγ together (Section 5). Their task is to approximate full softmax attention and the RoPE patterns of the teacher.

The ablation of the paper shows that stage 1 is important. On Llama-3-8B, MMLU decreases from 61.2 to 50.8 without stage 1 (Table 6).

### Important nuance

The paper did a sweep of learning rates that includes 1e-2 (Appendix B: 1e-2, 5e-3, 1e-3, 5e-4, 1e-4). Thus 1e-2 is not wrong by itself. But the final recipe of the paper uses 1e-3, and the stage 1 optimization of the runs is still materially different.

Thus the experiment is not "1e-2 must be bad". The experiment is: **run exactly the recipe of the paper, and see if the gap disappears.**

## 2. Stage 2: also very different

This is the second major gap in the reproduction.

| Setting | Paper | Runs of this project |
|---|---|---|
| LoRA targets | `q_proj`, `k_proj`, `v_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Rank | 8 | 8 |
| Alpha | 16 | 16 |
| Dropout | 0 | 0 |
| Learning rate | 5e-4 | 1e-4 |
| Schedule | Cosine with 10% warmup | ReduceLROnPlateau, no warmup |

The stage 2 config of the runs also freezes all Lizard parameters.

The paper specifies Wq, Wk and Wv, not Wo. The paper also reports that LoRA alone is sufficient. On the 8B model, full finetuning gets 61.4 on MMLU, and the default LoRA configuration gets 61.2 (Table 6).

### Test

Run four variants:

```text
A: current
B: paper stage-2 recipe, q/k/v only
C: paper recipe, q/k/v + Lizard params trainable
D: paper recipe, q/k/v only + Lizard params trainable
```

These variants separate the effect of the optimizer from the effect of the frozen Lizard parameters after stage 1.

**Open item:** In the review, variants C and D have the same description. Decide the difference between C and D before the runs start. Step 8 of the sequence at the end of this document compares q/k/v with q/k/v/o, and frozen with trainable Lizard parameters.

## 3. A real hole in the check of the checkpoint loading

This is the most important finding at the code level. The review recommends a fix immediately.

`load_model_for_eval.py` contains this code:

```python
_keys = model.load_state_dict(state_dict, strict=False)

assert len(_keys.unexpected_keys) == 0
```

But the code does **not** require that the expected trainable LoRA keys are present. Thus this message is misleading:

```text
*** All expected keys matched successfully ***
```

It means only: "The checkpoint has no unexpected keys." It does **not** mean that all LoRA keys loaded.

`src/model/load_model.py` has the same pattern. Links: [evaluation loader](../src/model/load_model_for_eval.py), [checkpoint loader](../src/model/load_model.py).

This is especially important because a checkpoint contains only the trainable weights, on purpose. Thus `missing_keys` always includes the frozen parameters of the base model. The check must look only at the trainable keys:

```python
trainable_keys = {
    n for n, p in model.named_parameters()
    if p.requires_grad
}

missing_trainable = trainable_keys.intersection(_keys.missing_keys)

assert not missing_trainable, (
    f"Missing trainable checkpoint weights: {sorted(missing_trainable)}"
)
assert not _keys.unexpected_keys
```

Do this check separately for stage 1 and for stage 2.

### Why this is important

If the LoRA keys did not load, the evaluated model was this combination:

- the base Llama,
- random or initial LoRA weights,
- the trained Lizard parameters.

This combination could give exactly the wide damage that the results show.

This is not necessarily the root cause. The stage 2 checkpoint is 3.49 MB, which agrees with ~1.7M LoRA parameters in bf16. But an experiment must exclude this cause. Do not assume that the load was successful.

**Repository check:**

- The code is in `src/model/load_model_for_eval.py` (lines 259–262) and in `src/model/load_model.py` (lines 61–65 and 137–141). When the checkpoint contains unexpected keys, the code prints an error message and continues.
- `save_trainable_weights` in `src/trainer/default_lm.py` saves only the parameters with `requires_grad`.
- A checkpoint key with a name that does not match a model parameter appears as an unexpected key. Thus the message shows that every key in the checkpoint loaded into the model. It does not show that every trainable parameter received a value from the checkpoint.
- [Document 6](06-evaluation-setup.md) says that the message appears "if all keys load". This description is not accurate. The message shows only that the checkpoint has no unexpected keys.

## 4. The "A" behavior on MMLU: a large clue

The MMLU result is not only 6.5 points lower. The answer-letter analysis ([document 7](07-results.md)) shows this behavior:

- If the model always gave "A", the macro average would be 23.1%.
- This "always A" accuracy has a correlation of +0.98 with the real accuracy per subject.
- After `Answer:`, the model in effect selects " A", independent of the question.

This is a very specific pathology.

The paper uses 5-shot MMLU. The harness scores the four choices with loglikelihood requests. Thus the choice token has a direct effect on the score.

But the review does **not** conclude that the harness itself is broken. On the MMLU subset, the scores are these:

| Model | Right answers | Accuracy |
|---|---|---|
| Teacher | 96 / 285 | 33.7% |
| Lizard model | 71 / 285 | 24.9% |

Thus the same harness gives a sensible result for the original teacher. For this reason, the MMLU evaluation itself is less suspicious than a degradation of the representations or the logits of the Lizard model.

### Possible causes of the "A" collapse

1. a poor approximation in stage 1,
2. collapse or saturation of the gate,
3. collapse of the feature maps,
4. a bias in the output or the logits, caused by the transformed attention,
5. a mismatch in the prompts or the tokenization,
6. a subtle difference in the scoring of `Answer:` → `" A"`,
7. numerical instability in the transformed attention.

Also, the PIQA and ARC-Easy results show that the problem is **not only a formatting problem of MMLU**.

## 5. Data and packing: another real gap in the reproduction

The data loader does not simply use "50K examples". It uses the first 200 examples for validation and all other examples for training (`src/dataloaders/alpaca_clean.py`):

```python
dataset = dataset['train']
train_set = convert_to_hf_dataset([dataset[ix] for ix in range(200, len(dataset))], cache_dir)
val_set   = convert_to_hf_dataset([dataset[ix] for ix in range(200)], cache_dir)
```

Then it concatenates the tokenized examples into chunks of 2048 tokens.

The paper used a curated subset of 50K cleaned Alpaca examples (Section 5). The notes report 51,560 training examples ([document 2](02-compute-and-cost.md)).

Thus compare the subset of the paper with the data that lolcats uses. Check these properties:

- the exact number of examples,
- the exact sequence of the examples,
- the split into training and validation data,
- the tokenizer,
- the handling of BOS,
- the handling of EOS,
- the prompt template,
- the boundaries of the packed sequences,
- the number of 2048-token sequences,
- the exact number of tokens that the model sees.

This is especially important for stage 1. In stage 1, the model learns the attention behavior of the teacher, including its behavior with positions. A different packing changes the distribution of positions that the gate sees.

The prompt construction also calls `tokenizer.encode(prompt, add_special_tokens=True)` explicitly. Then it adds the answer and EOS. At the end, it packs the examples together. Link: [Alpaca loader](../src/dataloaders/alpaca_clean.py).

Compare this code line by line with the preprocessing that the Lizard authors used. Do not assume that "the same dataset" means the same training data.

## 6. Numerical behavior can be different, although the equations match

This cause is easy to underestimate.

The checks in [document 8](08-verification.md) show that the equations and the wrapper give the expected results on small models. The checks include these parts:

- equivalence with the reference Lizard code,
- GQA,
- teacher mode,
- the cached path,
- recurrent decoding.

The review agrees with the notes: this is strong evidence against an obvious error in the algebra or in the connection into the model.

But it does **not** prove that training of the 1B model at full scale behaves numerically like the paper.

The paper introduces a hardware-aware GLA algorithm to solve a numerical instability (Section 4). This algorithm moves the gate contributions into the feature space.

This implementation calculates the dense form with gate products. It uses two methods for stability ([Lizard implementation](../src/model/linear_attention/lizard_attention.py)):

- `upcast`, which gives float32 or a more precise dtype,
- `.clamp_min(torch.finfo(...).tiny)` on the denominator of the gated branch.

These methods can be mathematically equivalent to the paper and still give different gradients and behavior at finite precision.

### Measurements

During stage 1, measure these values for each layer:

- the mean, standard deviation, minimum and maximum of γ,
- the fraction of γ values less than 1e-3,
- the fraction of γ values more than 0.999,
- the gate product over 128 tokens, over 256 tokens and over 512 tokens,
- the relative error ‖y_lizard − y_teacher‖ / ‖y_teacher‖.

These measurements give much more information than a direct step from the training loss to MMLU.

## 7. The stage 1 loss is possibly not equivalent to the loss of the paper

The LoLCATs implementation uses 1000 × mean squared error for each layer.

The paper describes stage 1 as an approximation of full softmax attention. But its hyperparameter table does not fully specify the reduction or the normalization inside the loss.

Thus measure these values:

- the raw MSE for each layer,
- the mean MSE,
- the sum MSE,
- the relative MSE.

Also compare the real gradient norms, not only the scalar loss.

This is important because the runs changed several settings at the same time:

- the normalization of the loss,
- the learning rate,
- β2 of the optimizer,
- the scheduler,
- the precision.

Thus a loss value of, for example, 0.01 does not show directly if stage 1 behaves as in the paper.

## 8. The evaluation harness is different

The notes say that this project uses an old pinned harness commit, and that the paper used a newer harness version ([document 6](06-evaluation-setup.md)). The harness configuration of this project also has these properties:

- It uses `hendrycksTest`.
- It calculates an unweighted mean over the 57 subjects.
- It adds no special tokens (no BOS).
- It evaluates with batch size 1.

Thus there is a real difference in the evaluation method.

But the review does not consider this difference the main explanation, for three reasons:

1. In the same harness, the teacher already gets a reasonable score.
2. PIQA and ARC-Easy also show significant damage.
3. On MMLU, the Lizard model is below chance and has a strong bias to "A".

Still, complete the missing teacher baselines:

- the teacher on all MMLU questions,
- the teacher on PIQA,
- the teacher on ARC-Easy.

Then compare the Lizard model with the teacher in this harness, not with the teacher of the paper. This comparison gives the real recovery loss in the harness of this project.

## 9. Check the evaluated checkpoint

Run 2 has a different subtle problem for reproducibility. The evaluated model came from this sequence:

1. the stage 1 checkpoint from Run 1,
2. a new stage 2 run after the NaN fix,
3. the evaluation of the stage 2 checkpoint.

Thus the evaluated model is not the model of the original run without interruption. This is reasonable. But record these values exactly:

- the SHA or hash of the stage 1 checkpoint,
- the SHA or hash of the stage 2 checkpoint,
- the best step,
- the final step,
- the validation loss.

The notes say that they do not record the final stage 2 validation loss of Run 2 ([document 4](04-training-runs.md)).

Also check that the evaluated `_ft.pt` is the **best** checkpoint, and not only the last checkpoint.

**Repository check:** The trainer writes the checkpoint file without a step suffix only when the validation loss improves (`src/trainer/default_lm.py`, lines 254–266). The periodic saves have the suffix `_<step>.pt`. Thus `..._ft.pt` is the checkpoint with the best validation loss. The file also contains the keys `step` and `eval/loss`. The hashes and the final step are not recorded yet.

## 10. Damage from stage 1 against damage from stage 2

This comparison shows where the problem starts. Evaluate three models:

- A. the original Llama-3.2-1B,
- B. the Lizard model after stage 1, before LoRA,
- C. the Lizard model after stage 2.

Use these tasks: the MMLU subset, PIQA and ARC-Easy.

The table shows how to read the outcome. The values are examples of MMLU-subset accuracy.

| Outcome | Teacher | After stage 1 | After stage 2 | Conclusion |
|---|---|---|---|---|
| Good result | 33.7 | 30–33 | 30–33 | Stage 2 has no problem. |
| Stage 1 failure | 33.7 | 24 | 23 | The attention approximation is the root problem. |
| Stage 2 failure | 33.7 | 31 | 23 | The root is clearly the finetuning. |

The review gives this experiment a higher priority than almost all other steps.

The tool is `scripts/compare_stages.sh`. The runs and their results are in [experiments/stage-difference.md](experiments/stage-difference.md).

## The strongest hypothesis

With all the evidence together, the review states this hypothesis:

> **The connection of the Lizard architecture into the model is probably right. But the LoLCATs recipe gives a much worse learned Lizard state than the recipe of the paper. The "A" collapse on MMLU is a downstream symptom.**

The reasons for this hypothesis:

- The implementation passes many reference tests and connection tests ([document 8](08-verification.md)).
- The disabled-branch evaluation on PIQA shows that both Lizard branches are active ([document 7](07-results.md)).
- The damage occurs on **PIQA, ARC-Easy and MMLU**. Thus it is not only a problem with long contexts.
- The paper states that the attention approximation stage is important (Table 6).
- The stage 1 recipe is different from the paper in five settings at the same time (Table 13). The five settings are **10× learning rate, scheduler, warmup, clipping and β2**.

## The debugging sequence

Do these steps in this sequence:

1. Check the trainable keys of the stage 1 checkpoint and the stage 2 checkpoint.
2. Evaluate the teacher on all MMLU questions, on PIQA and on ARC-Easy, in exactly the same harness.
3. Evaluate the checkpoint after stage 1 only.
4. Measure the relative MSE between the teacher and Lizard attention for each layer.
5. Run stage 1 again with the recipe of the paper.
6. Evaluate the checkpoint after stage 1 only again.
7. Run stage 2 with the recipe of the paper.
8. Compare q/k/v with q/k/v/o, and frozen with trainable Lizard parameters.
9. Examine the statistics of γ and of the gate decay.
10. Only after these steps, examine the details of the MMLU prompts and tokenization.

### Do not do this yet

Do **not** change the Lizard architecture yet. Examples are a larger window, RoPE in the student, one gate for each head, or different sinks.

The ablations of the paper show that w = 128, m = 4 is the good configuration. They also show that larger windows can cause a large decrease (Table 7). The default scalar gate is a deliberate choice (Table 4).

Thus the clean reproduction question is now this: **can the existing implementation recover the result of the paper with the recipe of the paper**? This experiment separates a problem in the implementation from a problem in the recipe.

## 11. Guideline to close the gap

### Two facts that shape the plan

- **Batching.** The jku-thesis pipeline packs Alpaca into 2048-token chunks, as lolcats does. Thus both pipelines take approximately 1,178 optimizer steps per stage. The learning rates of the two pipelines are directly comparable.
- **Configuration options of lolcats**. These settings are already config settings: the cosine schedule with warmup, the AdamW betas, the LoRA targets and trainable Lizard parameters in stage 2. Only gradient clipping needs a code change.

**Update from section 12:** Section 12 found a second necessary code change: float32 storage of the trainable weights. Section 12 puts this change before the new configs.

### Target and principle

**Target** (the paper, Table 9, Llama-3.2-1B): PIQA 74.8, ARC-Easy 65.6, MMLU 29.8. The Lizard model gets 68.0, 54.8 and 23.3.

**Principle:** Change one thing at a time. Check stage 1 before you spend money on stage 2.

### Phase 0: baselines (no training, the A10 is sufficient)

1. **Teacher on PIQA and ARC-Easy** in the harness of this project. Each evaluation takes a few minutes. If time is available, also evaluate the teacher on all MMLU questions. These evaluations give the real target in this configuration, because the paper used a newer harness.
2. **`gates.py` on the current Lizard model.** It records the gate behavior before any change, for the thesis.

### Phase 1: tools (small PRs, one change each)

| PR | Change | Reason |
|---|---|---|
| PR A | Optional gradient clipping in the trainer (`max_grad_norm`, disabled by default) | The paper clips at 1.0. lolcats cannot clip gradients now. |
| PR B | Two new configs with the recipe of the paper. They have new file names, so that no checkpoint gets overwritten. | See the table below. |
| PR C | A stage 1 check script. It measures the relative error ‖Ŷ − Y‖² / ‖Y‖² for each layer against the teacher, on held-out text. It compares this error with two reference points: untrained Lizard attention and the window branch alone. It also measures perplexity. | It shows in approximately one hour if stage 1 improved. |

The two configs in PR B:

| | Stage 1 config | Stage 2 config |
|---|---|---|
| Learning rate | 1e-3 | 5e-4 |
| Schedule | `cosine_warmup`, 118 warmup steps of 1,178 | Same as stage 1 |
| AdamW betas | (0.9, 0.99) | (0.9, 0.99) |
| Clipping | 1.0 | 1.0 |
| LoRA | – | r = 8, α = 16, on **q, k, v** (without o) |
| Lizard parameters in stage 2 | – | **Frozen**, as now |

The reason to keep the Lizard parameters frozen in this run: the paper does not say what it does. A change now would mix two effects. Phase 4 tests it separately.

**Start of a run with the new configs:** `make hf-job` does not give `DISTILL_CONFIG` and `FINETUNE_CONFIG` to the job. Put the configs in `ARGS` instead. There they replace the defaults, because argparse keeps the last value.

```bash
make hf-job HF_FLAVOR=h200 HF_TIMEOUT=6h ARGS="--distill_config <new_distill> --finetune_config <new_finetune>"
```

### Phase 2: one full run, but check stage 1 first (H200, approximately 4 h, approximately $19)

- **Check stage 1 while the job continues.** The stage 1 checkpoint reaches the Hub after approximately 50 minutes, and stage 2 continues. On the A10, run the PR C script on this checkpoint and on the current stage 1 checkpoint.
- **Decision point 1:** If the new stage 1 is **not** clearly better (lower error for each layer, lower perplexity), cancel the job. This saves approximately 3 hours. Then go to Phase 4.

### Phase 3: evaluate the finished model

- **Scores:** First PIQA and ARC-Easy (minutes). Then MMLU with `--limit 20` (1,140 questions, ±1.3, the same questions for every model). Then all MMLU questions.
- **Letter check:** Run `scripts/letters.py` on the log of the full MMLU evaluation. The correlation with one letter should disappear.
- **Checkpoint path:** In `eval.sh`, set `FT_CKPT` to the name of the full run (`...-bs=1-gas=8-nte=2-ms=-1-se=0-re=0_ft.pt`). The default of `FT_CKPT` is the name for a run of stage 2 only.
- **Decision point 2, success:** These three conditions are true:
  - PIQA and ARC-Easy are within approximately 1–2 points of the teacher *in this harness*.
  - MMLU is clearly above chance (≥ 27).
  - No single-letter pattern occurs.

  If the model gets there, the recipe was the cause. This is a clean result for the thesis.

**Repository check:** `eval.sh` builds the names of both checkpoints from `DISTILL_CONFIG` and `FINETUNE_CONFIG` (lines 15–24). Thus, for a run with the new configs, also set these two variables to the new config names.

### Phase 4: if a gap remains, try these steps in this sequence (least expensive first)

1. **Stage 2 with trainable Lizard parameters,** as the jku-thesis `train.py` does. This is one config line: `trainable_weights: [phi_q, phi_k, W_gamma, meta_tokens, alpha_blend]`.
2. **A sweep of the stage 1 learning rate.** The paper did a sweep over {1e-2, 5e-3, 1e-3, 5e-4, 1e-4} (Appendix B). Run stage 1 only for 2–3 values, and compare them with the PR C script. Each value costs approximately 50 minutes.
3. **A cross-check with the jku-thesis `train.py`.** It is an independent implementation of the recipe of the paper. If it gets the values of the paper and lolcats does not, compare the two pipelines:
   - the prompt template (`format_example` against the Alpaca format of LoLCATs),
   - the end-of-sequence tokens,
   - the tokens that count in the loss,
   - float32 against bf16.
4. **Only then consider architecture changes.** Check them against the ablations of the paper first (Tables 4, 6 and 7). At that point, they are a thesis contribution, not a fix.

**Expectation:** At 1B, even the Lizard model of the paper is only a little above chance on MMLU (29.8, against 31.0 for the teacher). The recovery will show most clearly on PIQA and ARC-Easy.

## 12. Contribution of each hyperparameter to the gap

**The gap:** PIQA −6.8, ARC-Easy −10.8 and MMLU −6.5 points below the Lizard 1B model of the paper. The damage is wide, and the checks found no error in the code. Thus the cause is on the training side.

A measurement changes the picture. The precision of the stored trainable weights decides if a learning rate works at all. Thus this section starts with the precision, and then examines each hyperparameter. Documents 1–10 do not describe this finding.

### Factor 0: weight precision (not in the recipe table, but it controls every learning rate)

The master weights are the stored copy of the trainable weights that the optimizer updates.

**Measured:**

- **No float32 master weights.** The lolcats trainer has no mixed precision: no autocast, and no float32 master copy of the weights. The Lizard parameters change to bf16 when the code builds the layer (`lizard_attention.py:149`).
- **LoRA is also bf16.** PEFT 0.9.0 casts the LoRA weights to the bf16 dtype of the base layer (`peft/tuners/lora/layer.py:118`).
- **The checkpoints agree:** 2.16 bytes per parameter in stage 1, and 2.05 in stage 2.

bf16 keeps only approximately 3 significant digits. An AdamW step moves each weight by approximately `lr × m/√v`. If the step is smaller than half the distance between neighboring bf16 numbers near the weight value, rounding discards it. The table shows the fraction of updates that rounding discards.

| Weights | Step (`m/√v`) | lr 1e-2 | 1e-3 | 5e-4 | 1e-4 |
|---|---|---|---|---|---|
| `alpha_blend` = 1.0 | Any | 0% | **100%** | **100%** | **100%** |
| φ weights after growth (~0.2) | Noisy (0.3) | 0% | 53% | 76% | 97% |
| LoRA A at initialization (±0.022) | Noisy (0.3) | 0% | 0% | 0% | **65%** |
| LoRA B after growth (~0.01) | Noisy (0.3) | 0% | 0% | 0% | **43%** |

**Consequences:**

- **Stage 1 at learning rate 1e-2 has no loss from rounding.** Rounding discards no updates. This is probably one reason for the high rate in the LoLCATs configs.
- **Stage 2 at learning rate 1e-4 loses approximately half of the noisy LoRA updates**. Thus LoRA learns even more slowly than its low rate suggests.
- **A change to the learning rates of the paper without float32 weights could make the results worse.**
  - At 1e-3, `alpha_blend` cannot change at all, and grown φ weights lose up to half of their updates.
  - The late part of a cosine schedule (down to 0.1×) would lose all of its updates.
  - The paper trained with FSDP-2. Its "bf16" most probably means the compute precision, with float32 master weights (standard for FSDP). But the paper does not say so explicitly.
- **The moment estimates of the optimizer are also bf16,** because the optimizer creates them with the dtype of the weights. This is a second loss of precision.

**Probable contribution:** large in stage 2, and a precondition for any change to the learning rate.

**Fix:** Keep the approximately 2M trainable parameters (Lizard parameters and LoRA) in float32, and keep the frozen model in bf16. This is a small code change. PEFT already casts the inputs to the LoRA dtype, and only `W_gamma` needs a cast of its input.

### Factor 1: stage 1 learning rate, 1e-2 against 1e-3 in the paper (10×)

- **Mechanism.** Adam moves each weight by approximately `lr` per step, independent of the gradient size. At a constant 1e-2 for 1,178 steps, a weight with an initial value of ~0.02 can drift by up to approximately 12. At 1e-3 with cosine decay, the maximum is approximately 0.6.
- **Three probable failure modes:**
  - **(a) The feature maps saturate.** With large φ weights, softmax(qW) becomes almost one-hot. The kernel loses its smoothness, and its gradients vanish.
  - **(b) The gate saturates.** With a large W_γ, γ = σ(W_γ x) stays near 0 or 1. Then the model loses the decay pattern, which is the only source of position information in Lizard attention.
  - **(c) The sink logits drift** until they take almost all of the attention, or none.
- **Evidence so far.** Stage 1 did learn: stage 2 started at perplexity 233, against 7,207 after the 10 steps of Run 0. Also, the disabled-branch evaluation shows that the gated branch is active. Thus the gated branch is not dead, but it may be partly saturated.
- **Testable prediction.** `gates.py` should show γ near 0 or 1 in many layers, with little variation from token to token.
- **Probable contribution:** medium to large. The quality of stage 1 limits all later results.

### Factor 2: schedule and warmup, constant without warmup against cosine with 10% warmup

- **No warmup.** With bias correction, the first Adam steps have full size (`m/√v ≈ ±1`) on new weights. Together with 1e-2, saturation most probably starts here.
- **No decay.** ReduceLROnPlateau with patience 10 evaluations gives at most one reduction in 1,178 steps. Thus the rate is almost constant. The final weights stay at the noise level of 1e-2 and do not settle. The selection of the best checkpoint by validation loss compensates only partly.
- **Probable contribution:** medium, mainly through stage 1. It amplifies factor 1.

### Factor 3: stage 2 learning rate, 1e-4 against 5e-4 in the paper (5×)

- **Mechanism.** In stage 2, only the LoRA adapters adapt the model to its new attention. A rate 5× lower, plus the bf16 losses of factor 0, makes the effective rate much lower still. The result is a model that did not fully adapt.
- **Evidence.** The model has damage even on short prompts. In Run 1, the training perplexity was still approximately 12–21 at step 386.
- **Check:** Compare the final stage 2 validation loss of Run 2 with the loss of the teacher on the same validation data. A large difference supports this factor.
- **Probable contribution:** large.

### Factor 4: Lizard parameters frozen in stage 2 (no information in the paper, trainable in jku-thesis)

- **Mechanism.** In stage 1, each layer learns from the inputs of the teacher (teacher forcing). In the model with Lizard attention, the inputs of a layer come from earlier Lizard layers. Thus the errors accumulate. End-to-end training of φ, γ, the sinks and α lets them adapt to this shift.
- **Probable contribution:** medium. Test it separately, after the recipe run. It needs the fix of factor 0, because at 5e-4 in bf16, rounding discards part of the φ updates.

### Factor 5: gradient clipping, none against 1.0

- **Mechanism.** With Adam, clipping is important mainly at spikes.
  - One very large gradient inflates the moment estimates. This causes one large step, then a long series of damped steps.
  - √v stays inflated for approximately 1/(1−β₂) steps. Clipping limits this effect.
  - Clipping does not prevent the `exp` overflow, which has a fix now. But it limits the spikes that can cause an overflow.
- **Evidence:** none in either direction. The stage 2 metrics never reached W&B, and the trainer does not log the gradient norm.
- **Probable contribution:** small to medium, mainly for stability.

### Factor 6: AdamW β₂, 0.999 against 0.99 in the paper

- **Mechanism.** β₂ sets the number of steps that the second-moment estimate averages over: approximately 1,000 against 100. The run has only 1,178 steps, and the gradient scale changes as the feature maps sharpen. Thus 0.999 lags. When the gradients decrease, the steps become too small. When the gradients increase, the steps become too large.
- **Probable contribution:** small alone, larger at learning rate 1e-2.

### Factor 7: LoRA targets, q, k, v, o against q, k, v in the paper

- **Mechanism.** The o target adds approximately 0.5M parameters of capacity. This is unlikely to *cause* a deficit. Table 8 of the paper shows little sensitivity to LoRA capacity (59.2–61.2 for ranks 4–64).
- **Probable contribution:** negligible. Change to q, k, v anyway, to match the paper.

### Factor 8: stage 1 loss scale, 1000 × mean MSE against summed squared error

- **Mechanism.** A constant scale factor on the loss has no effect on Adam (eps = 1e-8 is negligible here). Both forms give the layers equal weights.
- **Probable contribution:** none.

**Note:** This factor covers only loss forms that differ by a constant factor. Section 7 also lists the relative MSE, which is not such a form.

### Factor 9: checkpoint selection (not in the paper)

- **Mechanism.** Both stages keep the checkpoint with the best validation loss, and load it again at the end. The validation runs every 100 steps on 200 examples (approximately 20 sequences). If the validation is noisy, an early checkpoint can win.
- **Check (takes seconds):**

  ```python
  import torch
  for f in ['<..._distill.pt>', '<...-se=0-re=0-se=0-re=0_ft.pt>']:
      c = torch.load(f, map_location='cpu'); print(f[-40:], 'step', c['step'], {k: v for k, v in c.items() if 'loss' in k})
  ```

  If the stored step is far below 1,178, the evaluated model is an early checkpoint.
- **Probable contribution:** unknown until the check runs.

### Settings with no contribution

These settings match the paper: the data, the token count, the batch size and the sequence length. The window, the feature dimension and the number of sinks also match. A different seed changes PIQA by approximately one point at most, not by 7 points.

**Note:** Section 5 lists the data and the packing as a possible cause. The two sections disagree on this point. The checks in section 5 can decide it.

### Summary

| # | Factor | Stage | Probable size | Evidence so far | Least expensive test |
|---|---|---|---|---|---|
| 0 | bf16 master weights | Both stages | **Large** in stage 2. A precondition for factors 1–3. | Measured: rounding discards 43–65% of the LoRA updates at 1e-4. α is frozen at ≤ 1e-3. | Code change, then a stage 1 comparison |
| 3 | Stage 2 learning rate 5× lower | Stage 2 | **Large** | Wide damage. Training perplexity still ~12–21 at step 386. | Validation loss of the Lizard model against the teacher |
| 1 | Stage 1 learning rate 10× higher | Stage 1 | Medium to large | Stage 1 learned, but may be saturated. | `gates.py`, error for each layer |
| 2 | No warmup, no decay | Mainly stage 1 | Medium | The plateau scheduler is almost constant. | Error for each layer |
| 4 | Lizard parameters frozen in stage 2 | Stage 2 | Medium | – | Separate run |
| 9 | Selection of the best checkpoint | Both stages | Unknown | – | Stored `step` (seconds) |
| 5 | No clipping | Both stages | Small to medium | – | Log the gradient norm |
| 6 | β₂ 0.999 | Both stages | Small | – | – |
| 7 | LoRA also on o | Stage 2 | Negligible | Table 8 of the paper | – |
| 8 | Loss scale | Stage 1 | None | Adam does not change with the loss scale. | – |

### Changes to the plan in section 11

1. **Keep the trainable parameters in float32 before a run with the learning rates of the paper**. Without this change, a stage 1 rate of 1e-3 could make the results worse, and the comparison would mix two effects. This change becomes its own PR, before the configs of PR B.
2. **Separate the causes with an inexpensive experiment on stage 1 only.**
   - Run a 2 × 2 grid: {bf16, float32} weights × {current recipe, recipe of the paper}.
   - Each run takes approximately 50 minutes on the H200. Compare the runs with the PR C script.
   - For the thesis, this shows how much of the stage 1 gap comes from precision, and how much from the recipe. It gives this result before any stage 2 run.
3. **Two checks that can run now:** the stored checkpoint step (factor 9), and `gates.py` (the prediction of factor 1).

## 13. Ranking of the causes of the "A" collapse, and a debugging plan

This section ranks the seven possible causes of the "A" collapse in section 4. It also examines three possible root causes:

- the stage 1 hyperparameters,
- a mismatch between the code and the equations of the paper,
- the initialization of the Lizard parameters.

The task of stage 1 is the attention approximation. Thus the plan measures the attention approximation directly. It uses MMLU only as a final check.

**Basis:** No part of this section ran. The analysis uses three sources: the measurements of the [stage difference](experiments/stage-difference.md) and [temperature](experiments/temperature.md) experiments, the code, and the text of the paper (version 4). The mechanism in section 13.2 is a hypothesis.

### 13.1 Evidence that the ranking uses

- **MMLU subset:** the teacher gets 33.7. After stage 1, the Lizard model gets 22.5 and selects "A" for 95.4% of the questions. After stage 2, it gets 24.9 and selects "A" for 98.6%.
- **PIQA:** 57.6 after stage 1, and 68.0 after stage 2.
- **Gate:** in layers 1–15, γ is more than 0.999 for every token. Only layer 0 has a decay. Stage 2 does not change this.
- **Letter probability:** after `Answer:`, the letters " A" to " D" together get 1.8% of the next-token probability (stage 1, T = 1). Thus the model almost does not follow the 5-shot format.

### 13.2 A mechanism that connects the evidence (hypothesis)

1. **Sinks in the teacher.**
   - In Llama models, many heads from layer 1 on usually put most of their attention on the first token or on BOS. The value vector of such a sink token is near zero.
   - In layer 0, the attention is mostly local.
   - This project did not measure this for Llama-3.2-1B.
2. **The gated branch cannot make its output smaller.**
   - The code divides the gated branch by the sum of its weights (`src/model/linear_attention/lizard_attention.py:47`). Thus its output is always a weighted mean of value vectors.
   - α scales only the window branch. The sink logits occur only in the denominator of the window branch.
   - To copy a head that puts its attention on a sink, the gated branch must put its weight on the BOS tokens. These tokens have small value vectors.
   - In the packed Alpaca data, the BOS tokens are approximately 190 tokens apart (approximately 10M tokens per epoch over 51,560 examples). To reach them, the gate products must stay near 1. Thus γ goes to 1.
   - This agrees with the measured pattern: the gate decays in layer 0 and saturates in layers 1–15.
3. **No information about the order of the tokens.**
   - With γ ≈ 1 and no RoPE, the gated branch gets almost no information about the order of the tokens in the full prefix. The window branch gets almost none inside its 128 tokens.
   - Thus the model cannot connect "A." with the text of its option.
   - It also cannot copy the "Answer: X" pattern of the 5-shot examples. Copying needs information about the previous token.
   - This agrees with the letter probability of 1.8%. Then a prior for " A" decides the selection.
   - PIQA and ARC-Easy score the text of the answer, not a letter. Thus they lose less.
4. **BOS in training, no BOS in evaluation.**
   - Each training example starts with BOS (`src/dataloaders/alpaca_clean.py:132`).
   - The harness adds no special tokens for causal models (`lm_eval_harness/models_huggingface.py:379-380`).
   - If the gated branch uses BOS as its sink, the evaluation prompts give it no sink.

### 13.3 Ranking of the seven causes

| Rank | Cause (section 4) | Role | Evidence | Status |
|---|---|---|---|---|
| 1 | (2) Collapse or saturation of the gate | The concrete defect | γ > 0.999 in layers 1–15. Stage 2 does not fix it. | Measured |
| 2 | (1) A poor approximation in stage 1 | The general cause that contains causes 2 and 3 | The drop and the "A" collapse occur after stage 1 | This project measured the outcome, but not the error of each layer. |
| 3 | (4) A bias in the output or the logits | The path from the damaged attention to " A" | Letter probability 1.8%. "A" for 95–99% of the questions. | Hypothesis. A logit lens can test it. |
| 4 | (3) Collapse of the feature maps | A possible second defect | Weight RMS 0.02 → 0.15–0.22, maximum up to 2.3. The entropy of the features is not known. | Not measured |
| 5 | (5) A mismatch in the prompts or the tokenization | BOS in training, no BOS in evaluation | The teacher gets a sensible score in the same harness | A test of a few minutes |
| 6 | (7) Numerical instability | Mainly a problem of the training precision | The NaN has a fix. The Lizard calculations use float32. In 13 of 16 layers, the bf16 storage leaves the sink logits at exactly 0.5, 1 or 2. | Not a probable cause of the collapse |
| 7 | (6) A difference in the scoring of `Answer:` → `" A"` | – | Teacher 33.7 in the same harness. The temperature changes no prediction. | Least probable |

### 13.4 Three possible root causes

| Rank | Root cause | Evidence for | Evidence against | Least expensive test |
|---|---|---|---|---|
| 1 | Stage 1 hyperparameters, with the bf16 storage of the trainable weights | 10× learning rate, constant schedule, no warmup and no clipping. The new gate is the part that is most sensitive to these settings. In 13 of 16 layers, the sink logits stop at a power of two, which points to bf16 rounding. | LoLCATs trains well at 1e-2, but its attention has no gate | Single-layer bench (step 5 in section 13.5) |
| 2 | Mismatch between the code and the equations of the paper | Two ambiguous points that can have a large effect (below) | The code passes all reference tests | A review against the paper, then the single-layer bench |
| 3 | Initialization | W_γ = 0 gives γ = 0.5 at the start | A constant learning rate of 1e-2 can remove the effect of the start values in a few hundred steps | One arm of the single-layer bench |

#### Code against the equations of the paper

These parts of the code match the paper:

- one scalar gate for each token, shared by all heads (Section 5),
- the range of the gate products,
- a window of 128 tokens with 4 sink logits,
- the output Ŷ = Ŷ_gate + α · Ŷ_anchor, and the scale 1/√d,
- no RoPE.

At two points, the paper is not clear, and the code had to make a choice:

- **(a) Normalization of the gated branch.**
  - The parallel equation in Section 3.1 divides by the sum of the weights.
  - The recurrent form in the same section has no denominator. The matrix form in Section 4 also has no denominator.
  - The paper says that it uses FLA kernels. As far as this project knows, the GLA kernels of FLA do not normalize.
  - The code normalizes. A gated branch without normalization can make its output small by itself. Then it does not need the BOS tokens as sinks (section 13.2).
- **(b) Feature maps shared by all 32 heads.**
  - Each layer has one map from 64 to 128 dimensions for φq, and one for φk. These maps have 16,384 parameters for each layer.
  - The paper does not say if all heads share the feature maps. Appendix B says that the other design choices follow the defaults of LoLCATs.
  - The default of LoLCATs (`untied_head_einsum`) gives each head its own map. At feature dimension 128, this is 524,288 parameters for each layer.

Smaller points:

- The code uses Σ exp(tⱼ) in the denominator of the window branch. The paper writes Σ tⱼ ([document 9](09-paper-comparison.md)).
- The loss compares the outputs before `o_proj`.
- Section 4 writes the feature maps with `exp`. Table 13 gives softmax.

**Note:** The tests in [document 8](08-verification.md) compare the code with `reference.py` of jku-thesis. The same reading of the paper produced both. Thus these tests cannot find a wrong reading at point (a) or (b).

#### Initialization

- **W_γ = 0 gives γ = 0.5 at the start**. With this value, the gated branch keeps almost no weight on tokens more than one or two positions back.
  - The gradient from a token n positions back is proportional to approximately n · 0.5^(n−1). This is approximately 0.02 at n = 10, and approximately 1e-28 at n = 100.
  - Thus early training gets a signal only from near tokens. When γ increases, the signal from far tokens starts.
  - With a constant learning rate of 1e-2, no warmup and no clipping, the gate can then go directly to saturation.
  - GLA and Mamba-2 start most gate values much nearer to 1. The paper does not give a start value.
- **The start value of φ (standard deviation 0.02) is not a probable cause**. At feature dimension 128, the default initialization of LoLCATs gives a standard deviation of approximately 0.016.
- **α = 1 and sink logits = 0 are not probable causes**. With α = 1, the first output is approximately twice a normal attention output.

### 13.5 Debugging plan

The main metric is the error of the attention output of each layer against the teacher. A second metric is the similarity of the attention patterns. MMLU is only a final check. Steps 0–3 need no training. They run on the A10 with the existing checkpoints.

| Step | Question | Method (section of the [XAI document](experiments/xai.md)) | Cost |
|---|---|---|---|
| 0 | Does the missing BOS cause the collapse? | The MMLU subset and PIQA with and without BOS, for the teacher and the stage 1 model. Also the letter probability of the teacher. | A few minutes |
| 1 | How good is the approximation in each layer and each head? | Relative error ‖ŷ − y‖² / ‖y‖², with the inputs of the teacher and with the inputs of the Lizard model. Attention maps (section 9): the weight on the first token or BOS, the weight on the previous token, the mean distance. The weight of the gated branch on BOS. The entropy of the feature maps (section 8). A heatmap of the similarity of the hidden states (section 2). | Approximately 1 hour |
| 2 | Is γ ≈ 1 the best solution for the loss, or did the training stop there? | Set γ to a constant value in each layer (0.5, 0.9, 0.99, 0.999 or 1). Measure the stage 1 loss for each value. | Forward passes only |
| 3 | Which layers change the damaged attention into " A"? | A logit lens at `Answer:` (section 3). The KL divergence to the teacher (section 1). Replace the attention of one layer at a time with the exact teacher attention, with `train_attention=True` (section 10). A test that repeats a random token sequence, to find a loss of the order information. | Approximately 1 hour |
| 4 | Which reading of the paper is right? | List points (a) and (b) and the smaller points, each with the text of the paper | Desk work |
| 5 | Which root cause causes the saturation? | Record the inputs and outputs of the teacher for layers 0, 1, 8 and 15. The stage 1 of jku-thesis already trains layer by layer with hooks. Train only these layers, in float32. Combine: with or without normalization × shared or per-head φ × γ start 0.5 or near 1 × current recipe or recipe of the paper. | A few minutes for each run on the A10 |
| 6 | Does a full stage 1 with the fixes recover? | First keep the trainable weights in float32 (factor 0 of section 12). Then use the best settings from step 5. Record γ every 50 steps, and the gradient norm and update norm of each parameter group (section 7). | Approximately 50 minutes for each run on the H200 |
| 7 | Can stage 2 start? | Only after a stage 1 passes: a lower error in every layer, no saturated gate, a pass of the repeat test, and no single-letter pattern | – |

### 13.6 Decision rules

- **BOS removes the collapse (step 0):** the gated branch uses BOS as its sink. This supports point (a).
- **A forced decay gives a lower loss (step 2):** the training stopped at a bad point. Then examine the recipe or the start value of the gate.
  - This test prefers γ = 1, because the other parameters adapted to γ ≈ 1. Step 5 is the clean test.
- **Every variant with normalization saturates, and no variant without normalization saturates (step 5):** the normalization is the cause.
- **Saturation occurs only at a constant learning rate of 1e-2:** the hyperparameters are the cause.
- **Saturation occurs only with γ = 0.5 at the start:** the initialization is the cause.
- **Per-head feature maps give a much lower error:** the shared feature maps do not have sufficient capacity.
