# Experiment: a faster feedback loop

**Status:** A plan from 2026-10-09. Nothing has run, and the code changes of this plan do not exist yet. The plan comes from an analysis of an external text about fast debugging.

## Source

The text is the chapter "Debugging PyTorch programs" of Stas Bekman, [The Art of Debugging](https://github.com/stas00/the-art-of-debugging/tree/master/pytorch). It gives these ideas:

1. Make a tiny random model from the config of the real model, with a smaller hidden size, fewer layers and fewer heads.
2. Make a tiny tokenizer with a small vocabulary.
3. Make a tiny dataset with a few samples.
4. Load the real model with fewer layers, from the same weights.
5. Run the expensive attention in only some layers, and skip it in the other layers.
6. Measure the time and the memory with 2 and with 4 layers, and extrapolate to the full model.
7. Trim the lists that go with the number of layers, such as `layer_types`, together with it. A depth that is too small can remove a block type without an error.
8. Keep the script that makes a tiny model next to the model.

The text gives its own limit: "Unless you're testing the quality of a model, always use a tiny random model". A tiny random model finds errors in the code. It does not measure the quality.

## Question

Which ideas of the text give a faster and cheaper feedback loop for this project? The goal is a higher accuracy on MMLU, PIQA and ARC-Easy. Thus an idea is useful only if it does one of these two things:

- **Functional:** it finds an error in the code before an expensive run.
- **Quality:** it gives an accuracy signal earlier or for less money.

## The current loop

| Step | Machine | Time | Cost | Source |
|---|---|---|---|---|
| Setup of a job, evaluations every 100 steps, upload | H200 | ~10–15 min | ~$1 | Measured ([document 2](../02-compute-and-cost.md), [document 16](../16-optuna-plan.md)) |
| Stage 1, bf16 | H200 | ~50 min | ~$5 | Measured, Lizard Run 1 ([document 2](../02-compute-and-cost.md)) |
| Stage 1, float32 | H200 | 2.3–2.5 h | ~$12 | Measured, R1b and config 2 ([document 16](../16-optuna-plan.md)) |
| Stage 2, float32 | H200 | ~3 h 17 min | ~$17 | Measured ([stage 2 on config 1](stage2-config1.md)) |
| PIQA (1,838 questions) | A10 | 3 min 11 s | free | Measured ([document 16](../16-optuna-plan.md)) |
| ARC-Easy (2,376 questions) | A10 | ~8 min | free | Estimated ([document 16](../16-optuna-plan.md)) |

Thus each new idea costs one stage 1 run on the H200 before the first accuracy value: 1–2.5 hours and $5–12. The evaluation on the A10 is already cheap.

The gap is large. After stage 1, Lizard Run 1 has PIQA 57.6, and the LoLCATs attention has PIQA 73.5 in the same pipeline ([LoLCATs control](lolcats-control.md)). In MMLU, Lizard gives "A" for 95.4% of the questions, and LoLCATs for 19.3%. A change that closes a large part of this gap is visible with fewer questions and fewer layers.

## Analysis of each idea

| # | Idea of the text | Use in this project | Type | Decision |
|---|---|---|---|---|
| 1 | Tiny random model | A dry run of the full pipeline before each H200 job (step 2) | Functional | **Use** |
| 2 | Tiny tokenizer | None. A new vocabulary changes the token IDs. The Alpaca data cache and the harness then need new data. With the real vocabulary of 128,256 tokens and a hidden size of 512, the embedding has 66M parameters, small enough for a dry run. | – | Do not use |
| 3 | Tiny dataset | The trainer already has `--max_steps`, `--max_eval_batches` and `--eval_steps`. The CPU test of the [LoLCATs control](lolcats-control.md) used a small local Alpaca file. | Functional | **Use** the existing options. No new dataset on the Hub. |
| 4 | Real model with fewer layers | Stage 1 for the first N layers only. In stage 1, the result for these layers stays the same (next section). | Quality (MSE) | **Use** in step 4 |
| 5 | Expensive attention in some layers only | Teacher attention in all layers outside a set S, Lizard in S. Without training, from the existing checkpoints. | Quality (accuracy) | **Use first** (step 1) |
| 6 | Measure with 2 and 4 layers, then extrapolate | Time and memory for each Lizard layer on the A10 | Planning | **Use** in step 3 |
| 7 | Trim the lists that go with the depth | `softmax_attentions` holds layer indices. In a model with 4 layers, the index 15 does nothing, and no error occurs (`convert_attention`). The same applies to the sequence length and the window. At 256 tokens, the LoLCATs window covered the whole sequence, and the error was approximately 2e-13 ([LoLCATs control](lolcats-control.md), tests). | Functional | **Use** as checks in step 2 |
| 8 | Keep the script next to the tiny model | The script that makes the tiny model goes into `scripts/` | Reproducibility | **Use** |
| – | One process for the debugger | The project already uses one GPU and one process. | – | Nothing to do |
| – | Change the source code of `transformers` | The repository already changes its own classes at run time (`scripts/ablate.py`). The environment pins `transformers` 4.43.1. | – | Do not use |
| – | Replace attention with data of the right shape | This removes the quality signal. Idea 5 with the teacher attention keeps it. | – | Do not use |

## Why fewer layers keep the stage 1 result

- **Teacher input.** In stage 1, each Lizard layer gives the teacher output `y_true` to the next layer (`LolcatsLizardAttention.forward`). Thus the input of layer i depends only on the teacher layers before it. The Lizard parameters of the other layers have no effect on it.
- **Loss scale.** The loss is 1000 × the mean of the MSE of the trained layers (`distill_attention_xent_mse.py`). With fewer layers, the mean has fewer terms, so the gradient of each layer gets a larger factor. AdamW divides each update by the root mean square of the gradient. The weight decay is 0 in all stage 1 recipes for 1B. Thus this factor has almost no effect on the updates.
- **Exceptions.** Three parts couple the layers:
  - The recipe of the paper clips the global gradient norm at 1.0 (`max_grad_norm`). The LoLCATs recipe does not clip.
  - The plateau scheduler of the LoLCATs recipe and the selection of the best checkpoint use the mean validation loss of all trained layers.
  - bf16 rounding can differ.
- **Check.** Train layers 0–3 alone with the Run 1 config. Compare their MSE with the Run 1 values from [`fd128_lolcats.json`](xai-layer-wise-mse/fd128_lolcats.json): 0.1816, 0.4046, 0.4522 and 1.470. If they agree within a few percent, the shortcut is valid.

## Why the MSE alone is not sufficient

[Document 16](../16-optuna-plan.md) rejected a single-layer bench, because its score is the MSE. Over 6 Lizard runs, the stage 1 loss did not predict the accuracy. R1b had the lowest loss and the lowest ARC-Easy. But the LoLCATs control has a 7.3× lower loss and a 15.9 points higher PIQA ([XAI: layer-wise MSE](xai-layer-wise-mse.md), Run 2). Thus a very large drop of the MSE can come with a large gain in accuracy. This is one data point. Thus the plan uses the MSE only to reject candidates, and measures the accuracy of each remaining candidate (steps 1 and 4).

## The plan

### Step 1: teacher attention outside a set of layers (no training)

**Idea 5 of the text.** Load a stage 1 checkpoint. On each layer outside a set S, set `train_attention = True`. These layers then output the exact softmax attention with RoPE from their own q/k/v projections. Stage 1 does not change these projections, so this is the teacher attention ([XAI ideas](xai.md), section 10). Then evaluate PIQA and ARC-Easy.

| Set S (Lizard layers) | Purpose |
|---|---|
| No layer | Check of the patch: the result must equal the teacher (X0 of [document 19](../19-next-steps-from-literature.md)) |
| All 16 layers | The normal stage 1 model. It must give the known result (PIQA 57.6 for Lizard Run 1). |
| {0}, {15} | The layers with the largest differences to LoLCATs (36.9× and 17.5× in the MSE) |
| 0–3, 4–11, 12–15 | Three groups |
| All except 0, all except 15 | The gain from one teacher layer |

- **Checkpoints:** Lizard Run 1 stage 1 (fd128) and the [LoLCATs control](lolcats-control.md). The same sets for both give the gap for each group of layers.
- **Code:** a small script like `scripts/ablate.py`, with an environment variable for S. It must set the flag after the loader, because the loader sets `train_attention = False` ([document 6](../06-evaluation-setup.md)). The [LoLCATs layer](lolcats-control.md) has the same flag.
- **Cost:** 8 sets × 2 checkpoints × ~15 minutes, thus approximately 4 hours on the A10. Free. With `--limit 500`, less (step 5).
- **Decision:** if one group gives most of the drop, the next changes go to that group. This also answers the open question of [XAI: layer-wise MSE](xai-layer-wise-mse.md): which layer errors are most harmful for the accuracy.

### Step 2: dry run on a tiny random model (functional)

**Ideas 1, 3, 7 and 8 of the text.** Make a tiny Llama from the config of Llama-3.2-1B with `from_config`. This needs only the config file, not the weights.

| Setting | Tiny model | Llama-3.2-1B | Reason |
|---|---|---|---|
| Layers | 4 | 16 | Room for `softmax_attentions` tests |
| Hidden size | 512 | 2048 | |
| Heads, KV heads | 8, 2 | 32, 8 | The same ratio of 4, so the code for grouped KV heads runs |
| Head dimension | 64 | 64 | The feature maps keep their shape (64 → 128) |
| Intermediate size | 1024 | 8192 | |
| Vocabulary | 128,256 | 128,256 | The real tokenizer and the cached Alpaca data |
| dtype | bf16 | bf16 | The paper-LR run lost every update of α in bf16 ([paper LR](paper-lr.md)). A float32 test does not show this. |
| Sequence length | 512 | 2048 | Longer than the LoLCATs window (up to 255 tokens) and the Lizard window (128 + 4 sinks) |

- **Run:** stage 1 with `--max_steps 20 --gradient_accumulation_steps 1 --max_eval_batches 2 --eval_steps 10`. Then stage 2 with `--max_finetune_steps 10`. Then the harness (`lm_eval_harness/eval_lm_harness.py`) with `--limit 20`, and `layer_mse.py compute --max_batches 2`. `compare_stages.sh` has no option for `--limit` on PIQA and ARC-Easy. With 8 accumulation steps and few batches, no optimizer step can occur ([LoLCATs control](lolcats-control.md), tests).
- **Checks:**
  1. No crash and no NaN.
  2. Each trainable tensor changes after the steps, in bf16.
  3. The checkpoint loads with 0 missing and 0 unexpected keys.
  4. The validation loss decreases.
  5. Each index in `softmax_attentions` is smaller than the number of layers. This needs a new assert in `convert_attention`.
  6. The sequence length is longer than the window.
- **Machine and time:** the A10 or the CPU, minutes. Run it before each H200 job with a new option or a new script.
- **Code:** three new parts.
  - A script in `scripts/` that writes the tiny model to a local folder
  - A model config for each attention type
  - A make target `dry-run`
- **Limits:** the dry run does not measure the quality. It probably does not find errors that depend on the values of trained weights. An example is the NaN of Run 1 at step 386 of stage 2 ([document 5](../05-nan-crash.md)).

### Step 3: time and memory for each layer

**Idea 6 of the text.** Run stage 1 for 20 steps on the A10 with 1, 2 and 4 Lizard layers. The other layers stay softmax (`softmax_attentions`). Record the seconds for each micro-batch and the peak memory (`torch.cuda.max_memory_allocated`). The difference between the runs gives the cost of one Lizard layer. Extrapolate to 16 layers, and compare with the H200 speed (1.05–1.17 micro-batches each second in float32).

**Decision:** if a run with 1–4 Lizard layers fits on the A10 in a few hours, step 4 runs there for free. Otherwise, step 4 runs on the H200 for less time than a full run.

### Step 4: stage 1 of a candidate in a few layers

**Idea 4 of the text.** Train a candidate option only in a set S of layers. There are two methods:

- **`softmax_attentions`:** all layers outside S stay softmax and frozen (`convert_attention`). No code change. This works for each set S.
- **Truncation:** `num_hidden_layers: N` in the model config. `PretrainedModelLoader` passes unknown keys to `from_pretrained`, which then loads only the first N layers. This works only for S = 0 to N − 1, and it also saves the teacher layers after S. It needs a check.

Then measure two values:

1. The MSE of the layers in S (`layer_mse.py`).
2. The accuracy of the model with Lizard in S and the teacher elsewhere, as in step 1. Step 1 gives the result of v1 for the same S, without training.

- **First candidate:** `window_rope` ([document 13](../13-lizard-attention-v2.md)) in S = {0}, then in S = 0–3. Layer 0 has the largest difference to LoLCATs (36.9×). Finding 8 of the [LoLCATs control](lolcats-control.md) connects it with the window without RoPE.
- **Second candidate:** the same option in S = {15}.
- **Check of the method:** the v1 config in S = 0–3 must reproduce the Run 1 MSE of layers 0–3 (previous section).
- **Decision:** the candidate passes if its accuracy in S is higher than v1 by more than 2 SE (step 5). Only then does a full stage 1 run on the H200.

### Step 5: fewer questions for a screen

The harness has `--limit`. The binomial standard error (SE) is `sqrt(p (1 − p) / n)` ([document 7](../07-results.md)).

| Questions | SE at 50% | SE at 70% |
|---|---|---|
| 200 | 3.5 | 3.2 |
| 500 | 2.2 | 2.0 |
| 1,000 | 1.6 | 1.4 |
| 1,838 (all of PIQA) | 1.2 | 1.1 |
| 2,376 (all of ARC-Easy) | 1.0 | 0.9 |

- With 500 questions, the SE of a difference between two models is approximately 3 points. Thus a change that closes half of the PIQA gap (approximately 8 points) is visible. A change of 1–2 points needs all questions and more than one seed.
- Both models must get the same questions. The harness shuffles MMLU with a fixed seed ([document 7](../07-results.md)). For PIQA and ARC-Easy, compare the questions in the `write_out_info` files of the first screen.
- After stage 1, MMLU is near chance ([document 16](../16-optuna-plan.md)). The share of "A" answers in the MMLU subset is a larger signal: 95.4% for Lizard Run 1, 19.3% for LoLCATs.

## Order and expected savings

1. Step 1: no training and no cost. It shows which layers to change.
2. Step 2: one setup, then minutes before each H200 job.
3. Step 3: one hour or less on the A10. It decides where step 4 runs.
4. Step 4 for `window_rope`, with the screen of step 5.
5. A full stage 1 on the H200 only for candidates that pass, then all questions.

Each candidate that fails in step 4 saves one H200 run of 1–2.5 hours and $5–12. These are estimates from the measured times above. The speed of steps 3 and 4 is not measured yet.

## Open items

1. Write the script for step 1, and check it with S = no layer against the teacher.
2. Check that `num_hidden_layers` in the model config truncates the model, and that the loss of the first layers stays the same.
3. Check that `--limit` gives the same PIQA and ARC-Easy questions to each model.
