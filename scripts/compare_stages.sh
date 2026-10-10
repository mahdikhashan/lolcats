#!/usr/bin/env bash
# Find where the gap starts (docs/11-gap-analysis.md, section 10). Evaluates
#   A. the teacher, B. Lizard after stage 1 (no LoRA), C. Lizard after stage 2
# on the MMLU subset (5-shot, 5 questions per subject), PIQA and ARC-Easy (0-shot),
# and on request on all MMLU questions, ARC-Challenge, HellaSwag and WinoGrande (0-shot, as in Table 9 of the Lizard paper),
# on any machine with an NVIDIA GPU and conda (same setup as eval.sh)
# -> One-time setup: CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yaml
# -> conda activate lolcats-env
# -> export HF_TOKEN=<token with access to HF_REPO and meta-llama/Llama-3.2-1B>
# -> scripts/compare_stages.sh                                    # 3 models x 3 tasks
# -> MODELS="stage1 stage2" TASKS=piqa scripts/compare_stages.sh   # a part of it (TASKS can also include mmlu, all questions)
# -> MODELS="teacher stage2" TASKS="arc_challenge hellaswag winogrande" scripts/compare_stages.sh  # the other Table 9 tasks
# -> TEMPERATURE=0.5 scripts/compare_stages.sh                    # logits divided by 0.5 (see scripts/temperature.sh)
# -> MODELS=stage1 TASKS=layers scripts/compare_stages.sh          # no benchmark: only the checkpoint check and the
#    per-layer table (Lizard gates or LoLCATs window share), in a few minutes
# Paths below (OUT_DIR, checkpoints/, results/) are relative to the repo root; the script runs from there
# Writes OUT_DIR (default results/stages/<time>): env.txt, summary.md, summary.json, training/ with the
# training results CSVs from HF_REPO, and <model>/<task>/ with eval.log and the logs of compare_stages.py
# Override any variable below from the environment
set -euo pipefail
cd "$(dirname "$0")/.."

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
# The trainer writes the validation loss of each evaluation step next to each checkpoint, under results/
DISTILL_CSV=results/${DISTILL_CKPT#checkpoints/}; DISTILL_CSV=${DISTILL_CSV%.pt}.csv
FT_CSV=results/${FT_CKPT#checkpoints/}; FT_CSV=${FT_CSV%.pt}.csv

MODELS=${MODELS:-teacher stage1 stage2}
TASKS=${TASKS:-mmlu_subset piqa arc_easy}
LM_EVAL_DIR=${LM_EVAL_DIR:-$(cd .. && pwd)/lm-evaluation-harness}
LM_EVAL_COMMIT=b281b0921b636bc36ad05c0b0b0763bd6dd43463  # see lm_eval_harness/README.md
OUT_DIR=${OUT_DIR:-results/stages/$(date +%Y%m%d-%H%M%S)}
# Replaces the model config's cache_dir (/workspace/..., which may not be writable here)
CACHE_DIR=${CACHE_DIR:-${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}

echo '-> Checking the Python environment'
if ! python -c 'import torch, transformers, peft' 2>/dev/null; then
  echo 'Run this inside the lolcats-env conda env (see the top of scripts/compare_stages.sh)'
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

# The stage 2 checkpoint only when stage2 runs: a stage-1-only training has none
CKPTS=("$DISTILL_CKPT")
if [[ " $MODELS " == *" stage2 "* ]]; then CKPTS+=("$FT_CKPT"); fi
echo "-> Downloading checkpoints from $HF_REPO"
python - "$HF_REPO" "${CKPTS[@]}" <<'EOF'
import os, sys
from huggingface_hub import hf_hub_download
repo_id, *filenames = sys.argv[1:]
for filename in filenames:
    if os.path.isfile(filename):  # e.g., trained on this machine with make distill-local
        print(f'Using the local file {filename}')
        continue
    print(hf_hub_download(repo_id, filename, local_dir='.'))
EOF

mkdir -p "$OUT_DIR/training"
echo "-> Downloading the training results CSVs from $HF_REPO into $OUT_DIR/training, if they exist"
python - "$HF_REPO" "$OUT_DIR/training" "$DISTILL_CSV" "$FT_CSV" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
repo_id, out_dir, *filenames = sys.argv[1:]
for filename in filenames:
    try:
        print(shutil.copy(hf_hub_download(repo_id, filename, local_dir='.'), out_dir))
    except Exception as e:  # optional: the summary works without them
        print(f'Not available: {filename} ({type(e).__name__})')
EOF

echo "-> Recording the environment in $OUT_DIR/env.txt"
{
  echo "date (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "lolcats commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)$(git diff --quiet HEAD 2>/dev/null || echo ' (with uncommitted changes)')"
  echo "harness commit: $(git -C "$LM_EVAL_DIR" rev-parse HEAD)"
  echo "models: $MODELS"
  echo "tasks: $TASKS"
  echo "temperature: ${TEMPERATURE:-1}"
  echo "HF_REPO: $HF_REPO"
  echo "stage 1 checkpoint: $DISTILL_CKPT"
  echo "stage 2 checkpoint: $FT_CKPT"
  python -c 'import platform, torch, transformers, peft; print(f"python {platform.python_version()}, torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}, peft {peft.__version__}")'
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'GPU: nvidia-smi not available'
} | tee "$OUT_DIR/env.txt"

for model in $MODELS; do
  case $model in
    teacher) MODEL_ARGS=(--model_type model_config --model_config "$MODEL_CONFIG") ;;
    stage1)  MODEL_ARGS=(--model_type lolcats_ckpt --attn_mlp_checkpoint_path "$DISTILL_CKPT") ;;
    stage2)  MODEL_ARGS=(--model_type lolcats_ckpt --attn_mlp_checkpoint_path "$DISTILL_CKPT"
                         --finetune_checkpoint_path "$FT_CKPT") ;;
    *) echo "Unknown model $model (use teacher, stage1, stage2)"; exit 1 ;;
  esac
  for task in $TASKS; do
    COMMAND=eval
    case $task in
      mmlu_subset) TASK_ARGS=(--task hendrycksTest --num_shots 5 --limit 5) ;;  # 285 questions, see docs/07
      mmlu)        TASK_ARGS=(--task hendrycksTest --num_shots 5) ;;
      piqa)        TASK_ARGS=(--task piqa --num_shots 0) ;;
      arc_easy)    TASK_ARGS=(--task arc_easy --num_shots 0) ;;
      arc_challenge) TASK_ARGS=(--task arc_challenge --num_shots 0) ;;  # the paper reports acc_norm
      hellaswag)   TASK_ARGS=(--task hellaswag --num_shots 0) ;;       # the paper reports acc_norm
      winogrande)  TASK_ARGS=(--task winogrande --num_shots 0) ;;      # acc only
      layers)      TASK_ARGS=(); COMMAND=layers ;;  # the harness stops after the model load
      *) echo "Unknown task $task (use mmlu_subset, mmlu, piqa, arc_easy, arc_challenge, hellaswag, winogrande, layers)"; exit 1 ;;
    esac
    if [ "$COMMAND" = layers ] && [ "$model" = teacher ]; then
      echo "-> Skipping layers for the teacher: it has no Lizard or LoLCATs layers"; continue
    fi
    RUN_DIR=$OUT_DIR/$model/$task
    mkdir -p "$RUN_DIR"
    echo "-> Evaluating $model on $task, logging to $RUN_DIR/eval.log"
    LM_EVALUATION_HARNESS_PATH=$LM_EVAL_DIR LM_EVAL_RESULTS_PATH=$OUT_DIR/results_lm_eval.csv \
    PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH} python scripts/compare_stages.py "$COMMAND" "$RUN_DIR" \
      "${MODEL_ARGS[@]}" ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
      --cache_dir "$CACHE_DIR" --no_cache --no_wandb --verbose 2>&1 | tee "$RUN_DIR/eval.log"
  done
done

python scripts/compare_stages.py summary "$OUT_DIR"
cat "$OUT_DIR/summary.md"
