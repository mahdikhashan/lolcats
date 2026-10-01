# 11. Gap analysis

This document is a root cause analysis of the gap between the evaluation results of the Lizard model and the results of the paper. A review examined these sources:

- the notes in `docs/`,
- the Lizard implementation,
- the LoLCATs training configs,
- the code for evaluation and for model loading.

The gap looks **real**. The review does not treat it as mainly a problem of the MMLU evaluation.

Most statements in this document come from the review. A paragraph that starts with **Repository check** gives the result of a check against the code or the other documents. These checks occurred during the writing of this document.

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
