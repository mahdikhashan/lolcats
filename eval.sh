#!/usr/bin/env bash
# Evaluate the finetuned Lizard checkpoint on MMLU (5-shot) with LM Evaluation Harness,
# on any machine with an NVIDIA GPU and conda (no Docker or Hugging Face Jobs)
# -> One-time setup: CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yaml
# -> conda activate lolcats-env
# -> export HF_TOKEN=<token with access to HF_REPO and meta-llama/Llama-3.2-1B>
# -> ./eval.sh             # clones LM Eval, downloads the checkpoints from HF_REPO, evaluates
# -> ./eval.sh --limit 5   # extra args go to lm_eval_harness/eval_lm_harness.py (here, 5 questions per subject)
# Override any variable below from the environment, e.g., TASK=piqa NUM_SHOTS=0 ./eval.sh
set -euo pipefail
cd "$(dirname "$0")"

HF_REPO=${HF_REPO:-nanoman1/lolcats-lizard-llama-3.2-1b}
MODEL_CONFIG=${MODEL_CONFIG:-distill_llama3_2_1b_lizard_w128_fd128_m4}
DISTILL_CONFIG=${DISTILL_CONFIG:-distill_alpaca_clean_xent0_mse1000_lr1e-2_1b}
FINETUNE_CONFIG=${FINETUNE_CONFIG:-finetune_lora_qkvo_alpaca_clean_1b}
SEED=${SEED:-0}
REPLICATE=${REPLICATE:-0}

# Checkpoint paths in HF_REPO (and locally after downloading), as distill_llama.py names them
RUN=dl-d=$DISTILL_CONFIG-m=$MODEL_CONFIG-f=$FINETUNE_CONFIG-s=$SEED-se=$SEED-re=$REPLICATE
DISTILL_CKPT=${DISTILL_CKPT:-checkpoints/$MODEL_CONFIG/${RUN}_distill.pt}
# Finetuned with `make hf-job-finetune`; a full `make lizard` run ends in -bs=1-gas=8-nte=2-ms=-1-se=0-re=0_ft.pt instead
FT_CKPT=${FT_CKPT:-checkpoints/$MODEL_CONFIG/${RUN}-se=$SEED-re=${REPLICATE}_ft.pt}

TASK=${TASK:-hendrycksTest}  # MMLU in this LM Eval version
NUM_SHOTS=${NUM_SHOTS:-5}
LM_EVAL_DIR=${LM_EVAL_DIR:-$(cd .. && pwd)/lm-evaluation-harness}
LM_EVAL_COMMIT=b281b0921b636bc36ad05c0b0b0763bd6dd43463  # see lm_eval_harness/README.md
RESULTS_DIR=${RESULTS_DIR:-results/lm_eval}
# Replaces the model config's cache_dir (/workspace/..., which may not be writable here)
CACHE_DIR=${CACHE_DIR:-${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}

echo '-> Checking the Python environment'
if ! python -c 'import torch, transformers, peft' 2>/dev/null; then
  echo 'Run this inside the lolcats-env conda env (see the top of eval.sh)'
  exit 1
fi
python -c 'import torch; assert torch.cuda.is_available(), "PyTorch does not see a GPU"'
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
  echo 'Warning: no HF_TOKEN or saved Hugging Face login; downloading from a private HF_REPO or Llama will fail'
fi

echo "-> LM Evaluation Harness at $LM_EVAL_DIR"
if [ ! -d "$LM_EVAL_DIR/.git" ]; then
  git clone -q https://github.com/EleutherAI/lm-evaluation-harness "$LM_EVAL_DIR"
fi
if [ "$(git -C "$LM_EVAL_DIR" rev-parse HEAD)" != "$LM_EVAL_COMMIT" ]; then
  git -C "$LM_EVAL_DIR" checkout -q "$LM_EVAL_COMMIT"
fi
# Our model wrapper, which loads linearized checkpoints (see lm_eval_harness/README.md)
cp lm_eval_harness/models_huggingface.py "$LM_EVAL_DIR/lm_eval/models/huggingface.py"
python -m pip install -q -e "$LM_EVAL_DIR"

echo "-> Downloading checkpoints from $HF_REPO"
python - "$HF_REPO" "$DISTILL_CKPT" "$FT_CKPT" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo_id, *filenames = sys.argv[1:]
for filename in filenames:
    print(hf_hub_download(repo_id, filename, local_dir='.'))
EOF

mkdir -p "$RESULTS_DIR"
LOG=$RESULTS_DIR/$TASK-${NUM_SHOTS}shot-$(date +%Y%m%d-%H%M%S).log
echo "-> Evaluating $TASK ($NUM_SHOTS-shot), logging to $LOG"
LM_EVALUATION_HARNESS_PATH=$LM_EVAL_DIR LM_EVAL_RESULTS_PATH=$RESULTS_DIR/results_lm_eval.csv \
PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} python lm_eval_harness/eval_lm_harness.py \
  --model_type lolcats_ckpt \
  --attn_mlp_checkpoint_path "$DISTILL_CKPT" \
  --finetune_checkpoint_path "$FT_CKPT" \
  --cache_dir "$CACHE_DIR" \
  --task "$TASK" --num_shots "$NUM_SHOTS" --no_cache --no_wandb --verbose "$@" 2>&1 | tee "$LOG"

# The results CSV records MMLU as 0, so read the score from the log
grep 'MMLU RESULT' "$LOG" || true
