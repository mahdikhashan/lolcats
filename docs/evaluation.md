# Evaluation

Source: an answer from ChatGPT (2026-10-06), [link to the chat](https://chatgpt.com/c/6ac4f282-7694-83ed-9926-4d60fe78f9e0). This document keeps the answer without changes and does not check it. The numbers in its example tables are examples. They are not results of this project. [Document 7](07-results.md) gives the results of this project.

---

Yes. For your **Lizard thesis**, I would strongly recommend **not using WER as the main metric**. WER is primarily designed for speech recognition, where you have a known reference transcription. For an LLM generating free-form text, you need to evaluate several different things.

The key distinction is:

> **Are you trying to measure whether the generated text is similar to a reference, or whether the model still has the same capabilities after your architectural/training changes?**

For Lizard, the second one is much more important.

### 1. The evaluation stack I would use for your Lizard experiments

The Lizard paper itself evaluates whether the linearized model recovers the teacher's capabilities, using benchmarks such as MMLU and associative recall. ([arXiv][1])

I would structure your evaluation like this:

| Level                      | What you measure                              | Metrics                          | Useful for Lizard? |
| -------------------------- | --------------------------------------------- | -------------------------------- | ------------------ |
| **Language modeling**      | Does it model text well?                      | Perplexity                       | ⭐⭐⭐⭐⭐              |
| **Knowledge/reasoning**    | Does it retain capabilities?                  | MMLU, ARC, HellaSwag, PIQA       | ⭐⭐⭐⭐⭐              |
| **Generation quality**     | How good are generated answers vs reference?  | ROUGE, BLEU, BERTScore           | ⭐⭐⭐                |
| **Semantic similarity**    | Does student output preserve teacher meaning? | BERTScore / embedding similarity | ⭐⭐⭐⭐⭐              |
| **Teacher agreement**      | Does Lizard behave like original Llama?       | KL divergence / logit agreement  | ⭐⭐⭐⭐⭐              |
| **Exact task correctness** | Did it actually solve the problem?            | Accuracy / EM / Pass@k           | ⭐⭐⭐⭐⭐              |
| **Long-context behavior**  | Does linear attention preserve retrieval?     | Recall accuracy                  | ⭐⭐⭐⭐⭐              |
| **Efficiency**             | What did you gain?                            | latency, memory, throughput      | ⭐⭐⭐⭐⭐              |

The important point is that **there is no single "LLM generation quality" metric**.

---

# 2. If you specifically want to compare generated responses

Suppose you have:

```text
Prompt
   ↓
Teacher Llama-3.2-1B
   ↓
Teacher response

Prompt
   ↓
Your Lizard model
   ↓
Student response
```

You can save:

```json
{
  "prompt": "...",
  "teacher": "...",
  "student": "...",
  "config": "GLA+SWA",
  "seed": 42
}
```

Then compare teacher vs student.

### Metric 1 — Exact Match

Very strict:

```text
Teacher:  Paris
Student:  Paris
→ 1

Teacher:  Paris
Student:  Paris, France
→ 0
```

Useful for things such as QA and multiple-choice questions.

Not very useful for open-ended generation.

---

### Metric 2 — BLEU

Measures n-gram overlap.

Example:

```text
Teacher:
The cat is sitting on the table.

Student:
The cat sits on the table.
```

BLEU recognizes some lexical overlap.

But:

```text
Teacher:
The automobile is parked outside.

Student:
A car is standing outdoors.
```

could be semantically equivalent while having relatively poor BLEU.

So I wouldn't use BLEU as your primary Lizard metric.

---

### Metric 3 — ROUGE

Similar idea, commonly used for summarization.

For example:

```text
Teacher:
Transformers use self-attention to model relationships between tokens.

Student:
Self-attention allows Transformers to model relationships between tokens.
```

ROUGE will give a high similarity.

Again, this measures **surface-level similarity**, not necessarily correctness.

---

# 3. BERTScore is more interesting for your experiment

BERTScore compares contextual embeddings rather than simply matching exact words. The original BERTScore paper specifically proposes it for evaluating generated text and reports stronger correlation with human judgments than several traditional lexical metrics. ([arXiv][2])

Conceptually:

```text
Teacher:
The model is able to remember information from earlier tokens.

Student:
The network retains information from previous tokens.
```

BLEU/ROUGE:

```text
moderate similarity
```

BERTScore:

```text
high semantic similarity
```

For your **teacher vs Lizard student** comparison, this is much more meaningful.

---

# 4. But I would add embedding similarity

You can also calculate:

```text
cosine_similarity(
    embedding(teacher_output),
    embedding(student_output)
)
```

For example:

| Prompt |  Teacher |          Student | Cosine similarity |
| ------ | -------: | ---------------: | ----------------: |
| Q1     | answer A |         answer A |              0.98 |
| Q2     | answer B |     paraphrase B |              0.94 |
| Q3     | answer C | unrelated answer |              0.31 |

This is useful for answering a very specific thesis question:

> **Does my Lizard architecture preserve the semantic behavior of the original Transformer?**

---

# 5. Even better: compare the logits

For **your particular thesis**, I think this is potentially more interesting than comparing generated text.

You have:

```text
Original Llama
        ↓
     logits
        ↓
     Teacher

Lizard
        ↓
     logits
        ↓
     Student
```

For the same input, compare:

$$
P_T(y|x)
$$

against

$$
P_S(y|x)
$$

using KL divergence:

$$
D_{KL}(P_T || P_S)
$$

You could report:

| Configuration    |     MMLU |     PPL | BERTScore |     KL ↓ |
| ---------------- | -------: | ------: | --------: | -------: |
| Teacher          |     62.1 |     8.2 |      1.00 |        — |
| Lizard baseline  |     58.3 |    10.1 |      0.91 |     0.42 |
| Lizard + GLA     |     60.0 |     9.1 |      0.95 |     0.27 |
| Lizard + GLA+SWA | **61.2** | **8.6** |  **0.97** | **0.18** |

That tells a much more convincing story than simply:

> "The generated sentences look similar."

---

# 6. Perplexity is extremely important

For a base LLM such as your Llama model, calculate:

$$
PPL = \exp\left(
-\frac{1}{N}\sum_{i=1}^{N}\log P(x_i|x_{<i})
\right)
$$

Lower is better.

Example:

| Model   | WikiText PPL |
| ------- | -----------: |
| Teacher |      **8.2** |
| Lizard  |          9.1 |
| LoLCATs |         10.4 |

This tells you whether your model still behaves like a good language model.

Recent LLM evaluation work commonly reports perplexity alongside benchmark accuracy. ([lemma-public-asset.analemma.ai][3])

---

# 7. For Lizard, benchmark evaluation is probably the most important

Instead of creating your own arbitrary generation metric, use established benchmarks.

The current `lm-evaluation-harness` supports dozens of benchmarks including MMLU, HellaSwag, GSM8K and ARC, with reproducible configurations and metrics. ([LM Evaluation Harness][4])

For example:

```text
                 Teacher
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
     MMLU       HellaSwag      PIQA
       ↓            ↓            ↓
     62.1         78.4         76.2


                 Lizard
                    │
       ┌────────────┼────────────┐
       ↓            ↓            ↓
     MMLU       HellaSwag      PIQA
       ↓            ↓            ↓
     60.8         77.9         75.4
```

Then calculate:

$$
\Delta_{MMLU}
=
MMLU_{Lizard}-MMLU_{Teacher}
$$

and perhaps:

$$
Recovery =
\frac{Performance_{Lizard}}
{Performance_{Teacher}}
$$

This directly answers:

> **How much of the original Transformer capability did Lizard recover?**

That is very aligned with the Lizard paper's objective of near-lossless recovery. ([arXiv][1])

---

# 8. You can also evaluate generated answers with task-specific metrics

This is important because the correct metric depends on the task.

| Task                       | Recommended metric                  |
| -------------------------- | ----------------------------------- |
| Multiple choice            | Accuracy                            |
| QA with exact answer       | Exact Match                         |
| QA with flexible wording   | F1 / BERTScore                      |
| Summarization              | ROUGE / BERTScore                   |
| Translation                | BLEU / COMET                        |
| Speech recognition         | **WER**                             |
| Code generation            | Pass@1 / Pass@k                     |
| Mathematics                | Exact answer accuracy               |
| Language modeling          | Perplexity                          |
| Open-ended generation      | BERTScore + human/LLM evaluation    |
| Teacher-student comparison | KL divergence + semantic similarity |

So **WER isn't wrong**, but it is the wrong tool for most of your Lizard experiment.

---

# 9. A particularly good experiment for your thesis

Since you're changing the attention architecture/configuration, I would create a **fixed evaluation dataset**.

For example:

```text
                    1,000 prompts
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
        Teacher Llama             Lizard
             ↓                       ↓
       teacher outputs          student outputs
             │                       │
             └───────────┬───────────┘
                         ↓
                  Evaluation
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Semantic        Logits        Correctness
       similarity      similarity
          │              │              │
      BERTScore          KL           Accuracy
```

And **keep the prompts, decoding parameters and evaluation dataset identical**.

For every configuration:

```text
config A
config B
config C
config D
```

you get:

```text
PPL
MMLU
PIQA
HellaSwag
BERTScore
Embedding similarity
KL divergence
Generation latency
GPU memory
```

Then you can make a table like:

| Configuration |   PPL ↓ |   MMLU ↑ |   PIQA ↑ | BERTScore ↑ |    KL ↓ | Latency ↓ |   Memory ↓ |
| ------------- | ------: | -------: | -------: | ----------: | ------: | --------: | ---------: |
| Teacher       |     8.2 |     62.1 |     76.2 |       1.000 |       — |    100 ms |       8 GB |
| Lizard-A      |     9.8 |     58.4 |     73.1 |        .921 |     .42 | **61 ms** | **4.1 GB** |
| Lizard-B      |     9.1 |     60.2 |     74.9 |        .948 |     .29 |     64 ms |     4.3 GB |
| Lizard-C      | **8.7** | **61.1** | **75.8** |    **.971** | **.17** |     67 ms |     4.4 GB |

This would be **much more scientifically convincing** than evaluating only the generated text with WER.

---

## 10. One more important distinction for your thesis

I would actually separate your evaluation into **three experiments**:

### A. Capability preservation

> Does Lizard retain the capabilities of the original Llama?

Use:

**MMLU + PIQA + HellaSwag + ARC + etc.**

---

### B. Behavioral similarity

> Does Lizard behave similarly to the original Transformer?

Use:

**KL divergence + BERTScore + embedding similarity**

This is especially relevant to your teacher/student setup.

---

### C. Efficiency

> What do we gain by replacing softmax attention?

Measure:

**latency + throughput + peak VRAM + FLOPs + context-length scaling**

Then you can make the central thesis claim:

> **Lizard preserves model quality while reducing the computational/memory cost of attention.**

That is much closer to the scientific question than simply asking whether two generated sentences have similar wording.

The existing Lizard results follow essentially this philosophy: capability benchmarks plus long-context/associative-recall evaluation rather than a generic text-similarity score. ([arXiv][1])

If you want, I can next design a **concrete evaluation protocol for your Lizard 1B experiment**—including **exact datasets, number of samples, decoding settings, metrics, statistical tests, and a Python evaluation script** so that you can compare your different GLA/SWA configurations reproducibly.

[1]: https://arxiv.org/abs/2507.09025?utm_source=chatgpt.com "Lizard: An Efficient Linearization Framework for Large Language Models"
[2]: https://arxiv.org/abs/1904.09675?utm_source=chatgpt.com "BERTScore: Evaluating Text Generation with BERT"
[3]: https://lemma-public-asset.analemma.ai/online/fars/live/live_live_20260213/idea_ab465e86-006a-429e-a875-624acb53f090/main.pdf?utm_source=chatgpt.com "OUTPUT-SPACE"
[4]: https://lm-evaluation-harness.readthedocs.io/?utm_source=chatgpt.com "LM Evaluation Harness - LM Evaluation Harness"
