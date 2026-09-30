# 6. Evaluation setup

## Harness

The repository's evaluation script is `lm_eval_harness/eval_lm_harness.py`. It uses EleutherAI's
LM Evaluation Harness at commit **`b281b0921b636bc36ad05c0b0b0763bd6dd43463`** (the version the
Hugging Face Open LLM Leaderboard used at the time), as described in `lm_eval_harness/README.md`.
Facts about this version that matter for the results, checked in its source:

| Fact | Consequence |
|---|---|
| MMLU is called `hendrycksTest`: 57 tasks `hendrycksTest-<subject>`, data from `cais/mmlu` | Use `--task hendrycksTest`; `--task mmlu` finds no tasks |
| The script's MMLU score is the **unweighted mean over the 57 subjects** | It differs slightly from a per-question average |
| MMLU is scored as 4 loglikelihood requests per question (continuations " A" … " D") | 14,042 questions → about 56,000 requests |
| Few-shot examples come from each subject's `dev` split; `--limit N` takes the first N test questions after a shuffle seeded with 42 | `--limit` picks the **same questions for every model**, so runs can be compared question by question |
| `HuggingFaceAutoLM.__init__` calls `torch.set_grad_enabled(False)` | No gradients are kept during evaluation |
| Batches are right-padded and sorted longest first | An out-of-memory error would appear in the first minutes |
| The LoLCATs wrappers (`lm_eval_harness/models.py`) and the harness's own causal wrapper both return `add_special_tokens = False` | No beginning-of-text token is added for Llama, for either the teacher or Lizard; comparable between the two, but absolute scores can differ from other harness versions |
| `simple_evaluate` ignores `batch_size` when given a model object, and the script passes one | **Batch size is always 1**; `--batch_size` has no effect |

`lm_eval_harness/README.md` also asks to replace the harness's `lm_eval/models/huggingface.py` with
`lm_eval_harness/models_huggingface.py`. The patched file adds `cache_dir` and extra keyword
arguments; it still disables gradients. Earlier user commits (`805071c` and its predecessors)
filter the model config's keys down to those `HuggingFaceAutoLM.__init__` accepts.

### How the Lizard model is loaded for evaluation

`load_model_from_checkpoint` (in `src/model/load_model_for_eval.py`):

1. finds the model config from the checkpoint path (the folder name, or the `-m=` part of the file
   name) and the finetune config from the `-f=` part;
2. loads `LolcatsLlamaForCausalLM` through the harness wrapper;
3. swaps in Lizard attention and loads the stage 1 checkpoint, then sets `train_attention = False`;
4. wraps q/k/v/o with LoRA and loads the stage 2 checkpoint. It prints
   `*** All expected keys matched successfully ***` for each checkpoint if all keys load.

During evaluation `LolcatsLlamaModel` runs with `use_cache=True`, so every request goes through the
recurrent prefill path (`lizard_recurrent`), which computes the same result as the plain forward
([document 8](08-verification.md)).

## Pitfalls found, in order

1. **`ModuleNotFoundError: No module named 'src'`** (first eval attempt on HF Jobs, 2026-09-30
   02:16). `python lm_eval_harness/eval_lm_harness.py` puts `lm_eval_harness/` on the import path,
   not the repository root. Fix: run it with `PYTHONPATH=.`.
2. **Hard-coded `/workspace` paths.** `LM_EVALUATION_HARNESS_PATH`, the results CSV path, and the
   model config's `cache_dir` (`/workspace/lolcats/scratch/`, used to download the tokenizer and
   model) all point into `/workspace`, which a non-root user on a normal machine can't create.
   Fixed in PR #7 (see below).
3. **W&B entity in the eval script.** `eval_lm_harness.py` logs to the entity `hazy-research` by
   default. Always pass `--no_wandb`.
4. **The results CSV records MMLU accuracy as 0.** `save_results_to_dict` uses `mmlu_accs`, a local
   variable of `main()`; the resulting `NameError` is caught by a bare `except` that writes 0. The
   correct score is the printed line `MMLU RESULT: {'acc': ...}`. Other tasks are recorded correctly.
   Not fixed.
5. **The harness prints the base model name** (`'model': 'meta-llama/Llama-3.2-1B'`) for both the
   teacher and Lizard. The log file name or the command tells them apart.

Harmless messages during evaluation: the missing optional kernels (`causal_attention_cuda`,
ThunderKittens), `torch.load ... weights_only` warnings, `-> Using flash_attention_2 attention`
(every attention layer is replaced anyway), and
`Error at: 'LolcatsLlamaForCausalLM' object has no attribute 'layers' / But it's ok`.

## `eval.sh` (PR #7, commit `a9bfe49`)

Evaluates on any machine with an NVIDIA GPU and conda, without Docker or HF Jobs. Used on a
university machine with an A10 (24 GB).

One-time setup:

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

What it does:

1. checks that `torch`, `transformers` and `peft` import and a GPU is visible, and warns if no HF
   token is set;
2. clones LM Eval to `../lm-evaluation-harness` at `b281b09` (checking out only if the clone isn't
   already at that commit), copies in `models_huggingface.py`, and `pip install -e`s it for its
   dependencies;
3. downloads the stage 1 and stage 2 checkpoints from `HF_REPO` with `hf_hub_download`;
4. runs `eval_lm_harness.py` with `PYTHONPATH=.`, `--no_cache --no_wandb`, the Hugging Face cache as
   `--cache_dir`, and any extra arguments passed through; saves the output to
   `results/lm_eval/<task>-<shots>shot-<time>.log` and prints the `MMLU RESULT` line again at the end.

All settings (`HF_REPO`, `SEED`, `REPLICATE`, `DISTILL_CKPT`, `FT_CKPT`, `TASK`, `NUM_SHOTS`,
`LM_EVAL_DIR`, `RESULTS_DIR`, `CACHE_DIR`) can be overridden from the environment. By default
`FT_CKPT` is the finetune-only checkpoint (`...-se=0-re=0-se=0-re=0_ft.pt`).

Code changes in the same PR, so the script works outside `/workspace`:

- `eval_lm_harness.py` reads `LM_EVALUATION_HARNESS_PATH` and `LM_EVAL_RESULTS_PATH` from the
  environment (defaults unchanged), and passes `--cache_dir` to both model loaders.
- `load_model_from_checkpoint` and `load_model_from_config` take `cache_dir`, which replaces the
  model config's `cache_dir` when set.

## Teacher baseline

`--model_type model_config` reads only the `model:` section of a model config, so this evaluates
the unmodified Llama-3.2-1B with exactly the same harness settings:

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

Given an MMLU log, compares each subject's accuracy with how often each letter is the correct
answer in that subject's test split. If a model always answers one letter, its per-subject accuracy
equals that letter's frequency.

```bash
python letters.py results/lm_eval/hendrycksTest-5shot-<time>.log
```

### `ablate.py` (commit `ee8b8c4`)

Evaluates with one attention branch switched off at inference time, without retraining:

- `ABLATE=no_awa`: y = (1 + α) · GLA (window branch off)
- `ABLATE=no_gla`: y = (1 + α) · AWA (gated branch off)

Each branch alone is a weighted average of the values, so scaling by (1 + α) keeps the output at the
scale of GLA + α · AWA that the next layer expects. The script patches `LolcatsLizardAttention.lizard`
and routes the cached path to it (valid for loglikelihood tasks, not for generation), then runs
`eval_lm_harness.py` with the same arguments:

```bash
RUN=checkpoints/distill_llama3_2_1b_lizard_w128_fd128_m4/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2_1b-m=distill_llama3_2_1b_lizard_w128_fd128_m4-f=finetune_lora_qkvo_alpaca_clean_1b-s=0-se=0-re=0
ABLATE=no_gla LM_EVALUATION_HARNESS_PATH=../lm-evaluation-harness \
LM_EVAL_RESULTS_PATH=results/lm_eval/results_lm_eval.csv PYTHONPATH=. \
python ablate.py --model_type lolcats_ckpt \
  --attn_mlp_checkpoint_path ${RUN}_distill.pt --finetune_checkpoint_path ${RUN}-se=0-re=0_ft.pt \
  --cache_dir ~/.cache/huggingface/hub --task piqa --num_shots 0 --no_cache --no_wandb
```

Before use, the patch was tested on a tiny model: each mode gave exactly the expected branch output,
and the cached path used by the harness applied it too.

This measures how much the *trained* model relies on each branch. It is not the paper's ablation
(Table 6), which retrains without the branch.

### `gates.py` (not yet run)

Measures the trained gate values γ per layer on a real 5-shot MMLU prompt, and how much weight the
gated branch still gives to a token just outside the 128-token window:

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

Run with `PYTHONPATH=. python gates.py`. If "kept after 128 tokens" is near 0 in most layers, the
gated branch carries nothing past the window.
