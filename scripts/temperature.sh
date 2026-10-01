#!/usr/bin/env bash
# Temperature on the MMLU subset (docs/experiments/temperature.md): evaluate Lizard after stage 1 and
# after stage 2 on the MMLU subset (5-shot, 5 questions per subject) with the logits divided by each
# temperature, then compare the temperatures
# -> Same machine setup as scripts/compare_stages.sh: conda activate lolcats-env, export HF_TOKEN=...
# -> scripts/temperature.sh                                    # temperatures 0.1 0.5 1 2, stage1 and stage2
# -> TEMPERATURES="0.1 1" MODELS=stage2 scripts/temperature.sh  # a part of it (keep 1 as the reference)
# -> Select the GPU with CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 in front of the command
# Writes OUT_DIR (default results/temperature/<time>, relative to the repo root): T=<t>/ (one
# scripts/compare_stages.sh run each) and summary.md
set -euo pipefail
cd "$(dirname "$0")/.."

TEMPERATURES=${TEMPERATURES:-0.1 0.5 1 2}
MODELS=${MODELS:-stage1 stage2}
OUT_DIR=${OUT_DIR:-results/temperature/$(date +%Y%m%d-%H%M%S)}

for temperature in $TEMPERATURES; do
  echo "-> Temperature $temperature"
  TEMPERATURE=$temperature MODELS=$MODELS TASKS=mmlu_subset OUT_DIR=$OUT_DIR/T=$temperature scripts/compare_stages.sh
done

python scripts/compare_stages.py temperatures "$OUT_DIR"
cat "$OUT_DIR/summary.md"
