# References

This document lists external sources that the project notes use. Each entry gives a short description and its relation to this project.

| Item | Link | Description |
|---|---|---|
| Llama 3.2 evaluation details (Meta) | [meta-llama/llama-models, `models/llama3_2/eval_details.md`](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/eval_details.md) | The evaluation settings of Meta for the Llama 3.2 models. For MMLU with the pretrained models: 5-shot, the standard MMLU prompt, and the answer letter with the lowest negative log-likelihood. For the instruction-tuned models: the model generates the answer letter. Meta reports macro averages unless the text says otherwise. |
| llama-3.2-benchmark: MMLU evaluation notebook | [jaceroldan/llama-3.2-benchmark, `llama_32_b_mmlu_eval.ipynb`](https://github.com/jaceroldan/llama-3.2-benchmark/blob/main/llama_32_b_mmlu_eval.ipynb) | An attempt to replicate the MMLU result of Llama-3.2-1B-Instruct with Hugging Face `transformers`. Seven prompt variants give 15.7% to 42.9% accuracy over all questions. Run 7 uses the official evaluation prompts of Meta and gets 42.9% over all questions and 43.7% as the subject mean. |

## Notes

### Llama 3.2 evaluation details

The MMLU section of the document, without changes:

```text
### MMLU

For the pre-trained models we use a 5-shot config. To determine the choice character we use the standard MMLU prompt and compare the negative log-likelihood (NLL) of the various choices.

For the post-trained models we report both 5-shot and 0-shot scores. We ask the model to generate the best choice character. The 0-shot scores use a CoT (chain of thought) prompt. The maximum generation lengths for the 5-shot and 0-shot configs are 10 tokens and 1024 tokens respectively.

Macro averages are reported unless otherwise stated. The micro average scores for the various models are: 65.6, 79.0, and 85.4 for the pre-trained 8B, 70B and 405B models respectively for the 5-shot config; 69.44, 84.0, 87.71 for the post-trained 8B, 70B and 405B models respectively for the 5-shot config.
```

Relation to this project:

- **Pretrained models use the same kind of scoring as this project**. Meta compares the negative log-likelihood of the answer letters after a 5-shot prompt. It reports the macro average. The harness of this project at commit `b281b09` compares the log-likelihoods of " A" to " D" after a 5-shot prompt. It reports the unweighted mean over the 57 subjects, which is a macro average ([document 6](06-evaluation-setup.md)).
- **The reference value for the teacher is 32.2.** The [model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) gives MMLU 5-shot = 32.2 (`macro_avg/acc_char`) for the base model Llama-3.2-1B. The teacher of this project is this base model. In this harness, it got 33.7 on the MMLU subset ([document 7](07-results.md)).
- **Some details can still differ:** the exact prompt text, the few-shot examples, the tokenization, and the questions (all questions against the MMLU subset). Meta used its internal evaluation library. Meta published its evaluation data on Hugging Face, and the document links to it.
- **Instruction-tuned models use generation.** The model generates the answer letter, with at most 10 tokens in the 5-shot configuration. The model card gives MMLU 5-shot = 49.3 (`macro_avg/acc`) for Llama-3.2-1B-Instruct. The llama-3.2-benchmark notebook below replicates this generation method for the Instruct model.
- The micro averages in the document are for the 8B, 70B and 405B models only.
- The notes above come from `eval_details.md` and `MODEL_CARD.md` at commit `0e0b8c5` of `meta-llama/llama-models`, read on 2026-10-01.

### llama-3.2-benchmark

- The notebook evaluates **Llama-3.2-1B-Instruct**, not the base model `meta-llama/Llama-3.2-1B` of this project.
- It scores MMLU by text generation and a parse of the answer. This project scores MMLU with log-likelihoods of the answer letters, in the harness at commit `b281b09` ([document 6](06-evaluation-setup.md)). Thus its values are not directly comparable with the values of this project.
- It shows how much MMLU accuracy depends on the prompt and the answer extraction. The [XAI ideas](experiments/xai.md) refer to it for this reason (reference 4).
- Run 4 evaluated only 285 questions, not all 14,042, because its loop uses the size of the dev split.
- The notes above come from the notebook and the README at commit `958c0ad`, read on 2026-10-01.
