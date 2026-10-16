Lizard on Llama-3.2-1B
---

#### Description

A reproduction of Lizard ([arXiv:2507.09025](https://arxiv.org/abs/2507.09025)) on Llama-3.2-1B, built on the LoLCATs code base. Lizard replaces softmax attention with gated linear attention and a sliding window with sink tokens. Two distillation stages train it from the original model. This repository holds the training code, the evaluation scripts, the XAI scripts and the project notes of the master's thesis. The original LoLCATs README is in [README.lolcats.md](./README.lolcats.md).

#### Results

Accuracy in %. MMLU uses all 14,042 questions, 5-shot.

| Model | PIQA | ARC-Easy | MMLU |
| - | - | - | - |
| Teacher, this harness | 74.4 | 65.3 | 31.7 |
| First Lizard model | 68.0 | 54.8 | 23.3 |
| `window_rope`, stage 2 | 73.1 | 63.8 | 27.2 |
| Lizard 1B, paper | 74.8 | 65.6 | 29.8 |

---

#### Reproduce the thesis

All runs use seed 0. The checkpoints are public in the Hugging Face repository [`nanoman1/Llama-3.2-1B-Lizard`](https://huggingface.co/nanoman1/Llama-3.2-1B-Lizard), with a model card. During the work, its name was `nanoman1/lolcats-lizard-llama-3.2-1b`. The notes do not record the lolcats commit of the training runs.

| Result | Model config | Stage 1 / stage 2 config | Expected PIQA / ARC-Easy / MMLU | Eval commit |
| - | - | - | - | - |
| First Lizard model (Table 6.1, Figure 6.1) | `distill_llama3_2_1b_lizard_w128_fd128_m4` | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` / `finetune_lora_qkvo_alpaca_clean_1b` | 68.0 / 54.8 / 23.3 | not recorded |
| `window_rope` (Tables 9.4, 9.6, 9.7) | `distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope` | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` / `finetune_lora_qkvo_alpaca_clean_1b` | 73.1 / 63.8 / 27.2 | `84228c5`, MMLU `c132d29` |
| LoLCATs control (Table 7.4) | `distill_llama3_1_1b_lk_smd_wtk64_fd64_w01` | `distill_alpaca_clean_xent0_mse1000_lr1e-2_1b` / none | 73.5 / 62.8 / 26.0 (MMLU subset) | `8bd7614` |

Train on Hugging Face Jobs:

```bash
# first Lizard model, both stages
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=<hub repo> HF_FLAVOR=h200 HF_TIMEOUT=6h

# window_rope: stage 1, then stage 2 from its checkpoint
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=<hub repo> HF_FLAVOR=a100-large HF_TIMEOUT=6h \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope --no_finetune"
make hf-job-finetune IMAGE=mahdikhashan/lolcats HF_REPO=<hub repo> DISTILL_CKPT=<stage 1 checkpoint> \
  ARGS="--model_config distill_llama3_2_1b_lizard_v2_w128_fd128_m4_windowrope"

# LoLCATs control, stage 1 only
make hf-job IMAGE=mahdikhashan/lolcats HF_REPO=<hub repo> HF_FLAVOR=h200 HF_TIMEOUT=2h TARGET=lolcats ARGS="--no_finetune"
```

Evaluate on a GPU machine with conda:

```bash
CONDA_OVERRIDE_CUDA=12.4 conda env create -f environment.yaml
conda activate lolcats-env
export HF_TOKEN=...
MODELS="teacher stage2" TASKS="piqa arc_easy" scripts/compare_stages.sh
MODELS="teacher stage2" TASKS=mmlu scripts/compare_stages.sh
```

The commands of all other runs are in Appendix E of the [thesis](https://github.com/mahdikhashan/jku-thesis-drafts/releases/latest).

---

#### Documentation

- [Checkpoints on the Hugging Face Hub](https://huggingface.co/nanoman1/Llama-3.2-1B-Lizard)
- [Project notes](./docs/README.md)
- [Results](./docs/07-results.md)
- [Hyperparameters](./docs/hyperparameters.md)
- [Infrastructure](./docs/03-infrastructure.md)
- [Evaluation setup](./docs/06-evaluation-setup.md)
- [RoPE in the window branch](./docs/experiments/window-rope.md)
- [Scripts](./scripts/README.md)
- Related repositories
  - [jku-thesis](https://github.com/mahdikhashan/jku-thesis): the reference implementation and its tests
  - [jku-thesis-drafts](https://github.com/mahdikhashan/jku-thesis-drafts): the thesis

---

[LICENCE](./LICENSE)
