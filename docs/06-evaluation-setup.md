# 6. Evaluation setup

## Harness

The evaluation script of the repository is `lm_eval_harness/eval_lm_harness.py`. It uses the LM Evaluation Harness of EleutherAI at commit **`b281b0921b636bc36ad05c0b0b0763bd6dd43463`**, as `lm_eval_harness/README.md` specifies. The Hugging Face Open LLM Leaderboard used this version at that time. The table lists facts about this version that affect the results. Each fact comes from a check of the source code of the harness.

| Fact | Effect |
|---|---|
| The name of MMLU is `hendrycksTest`: 57 tasks `hendrycksTest-<subject>`, with data from `cais/mmlu`. | Use `--task hendrycksTest`. With `--task mmlu`, the harness finds no tasks. |
| The MMLU score of the script is the **unweighted mean over the 57 subjects**. | It is slightly different from the mean over all questions. |
| Each MMLU question gives 4 loglikelihood requests (continuations " A" … " D"). | 14,042 questions give approximately 56,000 requests. |
| The few-shot examples come from the `dev` split of each subject. `--limit N` takes the first N test questions after a shuffle with seed 42. | `--limit` selects the **same questions for every model**. Thus a comparison of two runs question by question is possible. |
| `HuggingFaceAutoLM.__init__` calls `torch.set_grad_enabled(False)`. | The evaluation keeps no gradients. |
| The harness pads batches on the right side and sorts them with the longest first. | An out-of-memory error, if any, occurs in the first minutes. |
| The LoLCATs wrappers (`lm_eval_harness/models.py`) and the causal wrapper of the harness both return `add_special_tokens = False`. | The harness adds no beginning-of-text token for Llama, for the teacher and for the Lizard model. Thus the two models are comparable. But the absolute scores can be different from scores of other harness versions. |
| `simple_evaluate` ignores `batch_size` when it receives a model object, and the script gives it a model object. | **The batch size is always 1.** `--batch_size` has no effect. |

`lm_eval_harness/README.md` also tells the user to replace `lm_eval/models/huggingface.py` of the harness with `lm_eval_harness/models_huggingface.py`. The patched file adds `cache_dir` and additional keyword arguments. It continues to disable gradients. Earlier commits of the user (`805071c` and the commits before it) filter the keys of the model config. They keep only the keys that `HuggingFaceAutoLM.__init__` accepts.

### How the evaluation loads the Lizard model

`load_model_from_checkpoint` (in `src/model/load_model_for_eval.py`) does these steps:

1. It finds the model config from the checkpoint path: the folder name, or the `-m=` part of the file name. It finds the finetune config from the `-f=` part.
2. It loads `LolcatsLlamaForCausalLM` through the harness wrapper.
3. It puts Lizard attention into the model and loads the stage 1 checkpoint. Then it sets `train_attention = False`.
4. It adds LoRA to q/k/v/o and loads the stage 2 checkpoint.

For each checkpoint, the function prints `*** All expected keys matched successfully ***` if all keys load.

During evaluation, `LolcatsLlamaModel` runs with `use_cache=True`. Thus each request goes through the recurrent prefill path (`lizard_recurrent`). This path gives the same result as the plain forward pass ([document 8](08-verification.md)).

## Problems found, in sequence

1. **`ModuleNotFoundError: No module named 'src'`** (first evaluation attempt on HF Jobs, 2026-09-30 02:16). The command `python lm_eval_harness/eval_lm_harness.py` puts `lm_eval_harness/` on the import path, not the root of the repository. Fix: run the script with `PYTHONPATH=.`.
2. **Fixed `/workspace` paths.** These paths all point into `/workspace`:
   - `LM_EVALUATION_HARNESS_PATH`,
   - the path of the results CSV,
   - the `cache_dir` of the model config (`/workspace/lolcats/scratch/`), where the tokenizer and the model download to.

   On a normal machine, a user without root access cannot create `/workspace`. PR #7 fixed this problem (see below).
3. **W&B entity in the evaluation script.** By default, `eval_lm_harness.py` logs to the entity `hazy-research`. Always give `--no_wandb`.
4. **The results CSV records the MMLU accuracy as 0.** `save_results_to_dict` uses `mmlu_accs`, which is a local variable of `main()`. The result is a `NameError`. A bare `except` catches this error and writes 0. The printed line `MMLU RESULT: {'acc': ...}` gives the true score. The CSV records the other tasks without errors. This problem has no fix yet.
5. **The harness prints the base model name** (`'model': 'meta-llama/Llama-3.2-1B'`) for the teacher and for the Lizard model. The name of the log file or the command shows which model ran.

These messages during evaluation do not show a problem:

- the missing optional kernels (`causal_attention_cuda`, ThunderKittens),
- the `torch.load ... weights_only` warnings,
- `-> Using flash_attention_2 attention` (Lizard replaces every attention layer, so this setting has no effect),
- `Error at: 'LolcatsLlamaForCausalLM' object has no attribute 'layers' / But it's ok`.

## `eval.sh` (PR #7, commit `a9bfe49`)

The script evaluates on any machine with an NVIDIA GPU and conda. It does not need Docker or HF Jobs. The evaluations in [document 7](07-results.md) ran with it on a university machine with an A10 (24 GB).

Preparation (one time only):

```bash
git clone https://github.com/mahdikhashan/lolcats && cd lolcats
CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yaml
conda activate lolcats-env
export HF_TOKEN=hf_...   # access to the checkpoint repo and to meta-llama/Llama-3.2-1B
```

Runs:

```bash
./eval.sh --limit 5                       # 5 questions per MMLU subject (285 questions)
./eval.sh                                 # full MMLU, 5-shot
TASK=piqa NUM_SHOTS=0 ./eval.sh           # other tasks
```

The script does these steps:

1. It checks that `torch`, `transformers` and `peft` import, and that a GPU is visible. It gives a warning if no HF token is available.
2. It clones the LM Evaluation Harness to `../lm-evaluation-harness` at `b281b09`. It runs `git checkout` only if the clone is not at that commit already. It copies `models_huggingface.py` into the clone. It runs `pip install -e` on the clone to install its dependencies.
3. It downloads the stage 1 checkpoint and the stage 2 checkpoint from `HF_REPO` with `hf_hub_download`.
4. It runs `eval_lm_harness.py` with these settings: `PYTHONPATH=.`, `--no_cache --no_wandb`, the Hugging Face cache as `--cache_dir`, and all additional arguments from the command line.
5. It saves the output to `results/lm_eval/<task>-<shots>shot-<time>.log`. At the end, it prints the `MMLU RESULT` line again.

Each setting accepts a value from the environment: `HF_REPO`, `SEED`, `REPLICATE`, `DISTILL_CKPT`, `FT_CKPT`, `TASK`, `NUM_SHOTS`, `LM_EVAL_DIR`, `RESULTS_DIR`, `CACHE_DIR`. By default, `FT_CKPT` is the stage 2 checkpoint of Run 2 (`...-se=0-re=0-se=0-re=0_ft.pt`).

The same PR changed the code, so that the script works outside `/workspace`:

- `eval_lm_harness.py` reads `LM_EVALUATION_HARNESS_PATH` and `LM_EVAL_RESULTS_PATH` from the environment. The default values did not change. The script also gives `--cache_dir` to both model loaders.
- `load_model_from_checkpoint` and `load_model_from_config` accept `cache_dir`. When it has a value, it replaces the `cache_dir` of the model config.

## Teacher baseline

`--model_type model_config` reads only the `model:` section of a model config. Thus the command below evaluates the unmodified Llama-3.2-1B with exactly the same harness settings:

```bash
LM_EVALUATION_HARNESS_PATH=../lm-evaluation-harness \
LM_EVAL_RESULTS_PATH=results/lm_eval/results_lm_eval.csv \
PYTHONPATH=. python lm_eval_harness/eval_lm_harness.py \
  --model_type model_config --model_config distill_llama3_2_1b_lizard_w128_fd128_m4 \
  --cache_dir ~/.cache/huggingface/hub \
  --task hendrycksTest --num_shots 5 --no_cache --no_wandb --limit 5
```

## Analysis scripts

### `letters.py` (commit `baa7b3e`)

The script reads an MMLU log. For each subject, it compares the accuracy with the frequency of each letter as the right answer in the test split of that subject. If a model always gives the same letter, its accuracy on a subject is equal to the frequency of that letter.

```bash
python letters.py results/lm_eval/hendrycksTest-5shot-<time>.log
```

### `ablate.py` (commit `ee8b8c4`)

The script evaluates with one attention branch disabled during inference. It does not retrain the model.

- `ABLATE=no_awa`: y = (1 + α) · GLA (window branch disabled)
- `ABLATE=no_gla`: y = (1 + α) · AWA (gated branch disabled)

Each branch alone gives a weighted average of the values. Thus the scale factor (1 + α) keeps the output at the scale of GLA + α · AWA, which the next layer expects. The script does these steps:

1. It patches `LolcatsLizardAttention.lizard`.
2. It sends the cached path to the patched function. This is valid for loglikelihood tasks, but not for generation.
3. It runs `eval_lm_harness.py` with the same arguments.

```bash
RUN=checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0
ABLATE=no_gla LM_EVALUATION_HARNESS_PATH=../lm-evaluation-harness \
LM_EVAL_RESULTS_PATH=results/lm_eval/results_lm_eval.csv PYTHONPATH=. \
python ablate.py --model_type lolcats_ckpt \
  --attn_mlp_checkpoint_path ${RUN}_distill.pt --finetune_checkpoint_path ${RUN}-se=0-re=0_ft.pt \
  --cache_dir ~/.cache/huggingface/hub --task piqa --num_shots 0 --no_cache --no_wandb
```

Before the evaluation runs, a test on a tiny model checked the patch. Each mode gave exactly the expected branch output. The cached path, which the harness uses, also applied the patch.

This method measures how much the *trained* model depends on each branch. It is not the ablation of the paper (Table 6). The paper trains a new model without the branch.

### `gates.py` (not run yet)

The script measures the trained gate values γ in each layer on a real 5-shot MMLU prompt. It also measures the weight that the gated branch gives to a token immediately outside the 128-token window.

```python
import os, torch
from datasets import load_dataset
from src.model.load_model_for_eval import load_model_from_checkpoint

run = 'checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0'
model, _, tok = load_model_from_checkpoint(f'{run}_distill.pt', f'{run}-se=0-re=0_ft.pt',
                                           cache_dir=os.path.expanduser('~/.cache/huggingface/hub'))

# A real 5-shot MMLU prompt (history questions include long passages)
subject = 'high_school_us_history'
ds = load_dataset('cais/mmlu', subject)
fmt = lambda d: d['question'] + ''.join(f'\n{l}. {c}' for l, c in zip('ABCD', d['choices'])) + '\nAnswer:'
prompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
prompt += ''.join(fmt(d) + f" {'ABCD'[d['answer']]}\n\n" for d in ds['dev']) + fmt(ds['test'][0])

gates = {}
for name, module in model.named_modules():
    if name.endswith('W_gamma'):
        layer = int(name.split('layers.')[1].split('.')[0])
        module.register_forward_hook(lambda m, i, o, layer=layer: gates.__setitem__(layer, torch.sigmoid(o.float())[0, :, 0]))

ids = tok(prompt, return_tensors='pt').input_ids.to(model.device)
with torch.no_grad():
    model(ids, use_cache=False)
print(f'{ids.shape[1]} tokens')
for layer, g in sorted(gates.items()):
    kept = g[-128:].log().sum().exp()  # weight left on a token just outside the window
    print(f'layer {layer:2d}: mean gamma {g.mean():.3f}  min {g.min():.3f}  max {g.max():.3f}  kept after 128 tokens {kept:.1e}')
```

To run the script, use `PYTHONPATH=. python gates.py`. If "kept after 128 tokens" is near 0 in most layers, the gated branch keeps no information from tokens beyond the window.
