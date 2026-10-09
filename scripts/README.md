# Scripts for evaluation and debugging

These scripts evaluate the Lizard model and help to find the cause of the gap to the paper. Training and the standard evaluation do not use them. The standard evaluation is `eval.sh` in the repository root.

| Script | Purpose | Documents |
|---|---|---|
| `compare_stages.sh` | Evaluates the teacher, the stage 1 model and the stage 2 model on the same tasks. Writes a summary with checkpoint checks and gate values. | [Gap analysis, section 10](../docs/11-gap-analysis.md), [Stage difference](../docs/experiments/stage-difference.md), [Feature dimension](../docs/experiments/feature-dimension.md) |
| `compare_stages.py` | The Python part of `compare_stages.sh` and `temperature.sh`: one evaluation with extra logs, the summary of a run, and the comparison of temperatures. | [Stage difference](../docs/experiments/stage-difference.md), [Temperature](../docs/experiments/temperature.md) |
| `temperature.sh` | Evaluates the stage 1 and stage 2 models on the MMLU subset with the logits divided by each temperature. | [Temperature](../docs/experiments/temperature.md) |
| `letters.py` | Compares the MMLU accuracy of each subject with the frequency of each answer letter, from the log of an MMLU evaluation. | [Document 6](../docs/06-evaluation-setup.md), [document 7, section 3](../docs/07-results.md) |
| `ablate.py` | Evaluates the Lizard model with the gated branch or the window branch disabled during inference. | [Document 6](../docs/06-evaluation-setup.md), [document 7, section 5](../docs/07-results.md), [XAI ideas, section 5](../docs/experiments/xai.md) |
| `layer_mse.py` | Calculates the MSE between the teacher attention and the Lizard attention for each layer, from a stage 1 checkpoint. Plots the layers of several checkpoints, as Figure 14 of the LoLCATs paper. | [XAI: layer-wise MSE](../docs/experiments/xai-layer-wise-mse.md) |
| `attention_weights.py` | Calculates the attention weights of the teacher and of Lizard (stage 1, stage 2 or the initial weights) on packed Alpaca samples. Plots them as Figures 18–21 of the LoLCATs paper, with metrics for all heads. | [XAI: sample attention weights](../docs/experiments/xai-sample-attention-weight.md) |
| `position_mse.py` | Calculates the MSE of `layer_mse.py` for each bucket of query positions (0–127, 128–511, 512–2047). Ranks several checkpoints by the MSE of each bucket against their PIQA and ARC-Easy accuracy. Draws heatmaps of layers × buckets. | [XAI: MSE by token position](../docs/experiments/xai-position-mse.md) |

## How to run

Run the Python scripts from the repository root. The shell scripts change to the repository root, so they run from any directory. Relative paths, such as `OUT_DIR`, are relative to the repository root.

```bash
scripts/compare_stages.sh                                   # teacher, stage 1 and stage 2 on 3 tasks
MODELS=stage1 TASKS=mmlu_subset scripts/compare_stages.sh   # a part of it
scripts/temperature.sh                                      # temperatures 0.1, 0.5, 1 and 2
python scripts/compare_stages.py summary results/stages/<time>
python scripts/letters.py results/lm_eval/hendrycksTest-5shot-<time>.log
ABLATE=no_gla PYTHONPATH=. python scripts/ablate.py <arguments of lm_eval_harness/eval_lm_harness.py>
python scripts/layer_mse.py compute results/layer_mse/<name>.json --model_config <model config> --distill_config <distill config>
python scripts/layer_mse.py plot results/layer_mse/<plot name> results/layer_mse/<name>.json ...
python scripts/attention_weights.py compute results/attention_weights/<name>.json --model_config <model config> --distill_config <distill config> [--stage 2]
python scripts/attention_weights.py plot results/attention_weights/<plot name> results/attention_weights/<name>.json ...
python scripts/position_mse.py compute results/position_mse/<name>.json --from_json docs/experiments/xai-layer-wise-mse/<name>.json
python scripts/position_mse.py rank results/position_mse/<name> results/position_mse/*.json --accuracy docs/experiments/xai-position-mse/accuracy.csv
python scripts/position_mse.py heatmap results/position_mse/<plot name> results/position_mse/<name>.json ...
```

The comments at the top of each script give all options. The shell scripts, `ablate.py`, `layer_mse.py compute`, `attention_weights.py compute` and `position_mse.py compute` need the `lolcats-env` conda environment and a GPU. `compare_stages.sh` explains the setup.
