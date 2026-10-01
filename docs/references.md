# References

This document lists external sources that the project notes use. Each entry gives a short description and its relation to this project.

| Item | Link | Description |
|---|---|---|
| llama-3.2-benchmark: MMLU evaluation notebook | [jaceroldan/llama-3.2-benchmark, `llama_32_b_mmlu_eval.ipynb`](https://github.com/jaceroldan/llama-3.2-benchmark/blob/main/llama_32_b_mmlu_eval.ipynb) | An attempt to replicate the MMLU result of Llama-3.2-1B-Instruct with Hugging Face `transformers`. Seven prompt variants give 15.7% to 42.9% accuracy over all questions. Run 7 uses the official evaluation prompts of Meta and gets 42.9% over all questions and 43.7% as the subject mean. |

## Notes

### llama-3.2-benchmark

- The notebook evaluates **Llama-3.2-1B-Instruct**, not the base model `meta-llama/Llama-3.2-1B` of this project.
- It scores MMLU by text generation and a parse of the answer. This project scores MMLU with log-likelihoods of the answer letters, in the harness at commit `b281b09` ([document 6](06-evaluation-setup.md)). Thus its values are not directly comparable with the values of this project.
- It shows how much MMLU accuracy depends on the prompt and the answer extraction. The [XAI ideas](experiments/xai.md) refer to it for this reason (reference 4).
- Run 4 evaluated only 285 questions, not all 14,042, because its loop uses the size of the dev split.
- The notes above come from the notebook and the README at commit `958c0ad`, read on 2026-10-01.
