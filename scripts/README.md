# Scripts for evaluation and debugging

These scripts evaluate the Lizard model and help to find the cause of the gap to the paper. Training and the standard evaluation do not use them. The standard evaluation is `eval.sh` in the repository root.

| Script | Purpose | Documents |
|---|---|---|
| `compare_stages.sh` | Evaluates the teacher, the stage 1 model and the stage 2 model on the same tasks. Writes a summary with checkpoint checks and gate values. | [Gap analysis, section 10](../docs/11-gap-analysis.md), [Stage difference](../docs/experiments/stage-difference.md), [Feature dimension](../docs/experiments/feature-dimension.md) |
| `compare_stages.py` | The Python part of `compare_stages.sh` and `temperature.sh`: one evaluation with extra logs, the summary of a run, and the comparison of temperatures. | [Stage difference](../docs/experiments/stage-difference.md), [Temperature](../docs/experiments/temperature.md) |
| `temperature.sh` | Evaluates the stage 1 and stage 2 models on the MMLU subset with the logits divided by each temperature. | [Temperature](../docs/experiments/temperature.md) |
| `letters.py` | Compares the MMLU accuracy of each subject with the frequency of each answer letter, from the log of an MMLU evaluation. | [Document 6](../docs/06-evaluation-setup.md), [document 7, section 3](../docs/07-results.md) |
| `ablate.py` | Evaluates the Lizard model with the gated branch or the window branch disabled during inference. | [Document 6](../docs/06-evaluation-setup.md), [document 7, section 5](../docs/07-results.md), [XAI ideas, section 5](../docs/experiments/xai.md) |

## How to run

Run the Python scripts from the repository root. The shell scripts change to the repository root, so they run from any directory. Relative paths, such as `OUT_DIR`, are relative to the repository root.

```bash
scripts/compare_stages.sh                                   # teacher, stage 1 and stage 2 on 3 tasks
MODELS=stage1 TASKS=mmlu_subset scripts/compare_stages.sh   # a part of it
scripts/temperature.sh                                      # temperatures 0.1, 0.5, 1 and 2
python scripts/compare_stages.py summary results/stages/<time>
python scripts/letters.py results/lm_eval/hendrycksTest-5shot-<time>.log
ABLATE=no_gla PYTHONPATH=. python scripts/ablate.py <arguments of lm_eval_harness/eval_lm_harness.py>
```

The comments at the top of each script give all options. The shell scripts and `ablate.py` need the `lolcats-env` conda environment and a GPU. `compare_stages.sh` explains the setup.
