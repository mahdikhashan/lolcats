Yes. I found a much richer set of papers in this direction. **Several are unusually close to what you are doing with Lizard** because they reproduce a published model, discover a discrepancy, and then investigate *why* the discrepancy occurred.

I would divide them into four groups.

## 1. Closest matches to your Lizard reproduction problem

| Paper                                                                                                       | What they reproduced                         | What went wrong / what they discovered                                                                                                                                  | Methodology                                                                                                                               | Relevance to Lizard                                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **The MultiBERTs: BERT Reproductions for Robustness Analysis** (Sellam et al., ICLR 2022)                   | 25 independently trained BERT-base models    | A single pretrained checkpoint does not necessarily represent the general training procedure. Random initialization and data shuffling can produce meaningful variation | Re-trained BERT with controlled hyperparameters while varying initialization and data order; evaluated multiple checkpoints statistically | **★★★★★ Extremely relevant.** Don't rely on one Lizard run. Test multiple seeds and distinguish *architecture effects* from *training stochasticity*. ([arXiv][1])          |
| **Reproducibility Issues for BERT-based Evaluation Metrics** (Chen et al., EMNLP 2022)                      | Four published BERT-based evaluation metrics | Results failed to reproduce because of undocumented preprocessing, missing code, weak baselines, and even correlation with the wrong CSV column                         | Reimplemented the methods, traced discrepancies, then performed controlled preprocessing experiments                                      | **★★★★★** Almost a blueprint for your "why is my MMLU 23% instead of 32%?" investigation. ([ACL Anthology][2])                                                              |
| **A Systematic Review and Replicability Study of BERT4Rec** (Petrov & Macdonald, RecSys 2022)               | BERT4Rec                                     | Default configuration did **not** reproduce the paper; original results could be obtained only after training up to **30× longer**                                      | Reviewed subsequent papers, tested original implementations/configs, then created an independent HF implementation                        | **★★★★★** Excellent example of discovering that the apparent architecture/model gap was actually substantially affected by the training procedure. ([arXiv][3])             |
| **Replicability under Near-Perfect Conditions – A Case-Study from Automatic Summarization** (Mieskes, 2022) | Published summarization experiments          | Even with seemingly complete artifacts, results were not automatically reproducible; missing experimental details still mattered                                        | Attempted replication under unusually favorable conditions and systematically examined factors affecting replication                      | **★★★★★** Very good precedent for documenting every Lizard implementation detail rather than assuming "the code is available, therefore reproducible." ([ACL Anthology][4]) |
| **MultiBERTs BERT Reproductions for Robustness Analysis**                                                   | BERT pretraining                             | Conclusions based on a single model can be conclusions about that *specific checkpoint*, not the architecture/training procedure                                        | 25 full models + 140 intermediate checkpoints                                                                                             | **★★★★★** Particularly useful for your planned XAI/learning-dynamics analysis. ([Google Research][5])                                                                       |

---

# 2. Papers specifically showing that implementation details can change the result

These are particularly useful for your **Lizard MMLU gap investigation**.

| Paper                                                        | Finding                                                                                                                                    | Why it matters for you                                                                                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Reproducibility Issues for BERT-based Evaluation Metrics** | Undocumented preprocessing alone could substantially alter results; one reported result was inflated because the wrong CSV column was used | Your **MMLU evaluation pipeline needs to be treated as part of the experiment**, not an afterthought. ([ACL Anthology][2])                                    |
| **A Systematic Review and Replicability Study of BERT4Rec**  | Default training configuration failed to reproduce the original result; substantially longer training recovered it                         | Check **training tokens/steps, LR schedule, warmup, convergence and checkpoint selection** before blaming GLA/SWA. ([arXiv][3])                               |
| **Replicability under Near-Perfect Conditions**              | Having source code/data/artifacts was still insufficient                                                                                   | Your reproduction table should include *every* implementation assumption: masking, initialization, tokenizer, precision, optimizer, etc. ([ACL Anthology][4]) |
| **MultiBERTs**                                               | Random initialization and data order materially affect conclusions                                                                         | Run several seeds before deciding that the architectural modification caused the MMLU gap. ([arXiv][1])                                                       |
| **A Primer in BERTology**                                    | Reviews evidence that BERT fine-tuning can vary substantially with initialization and data order                                           | Useful background citation for arguing that a single fine-tuning/reproduction result isn't sufficient. ([DOI][6])                                             |

---

# 3. Papers that directly study reproducibility of NLP experiments

There is a whole literature here that is worth mining for your thesis methodology.

### **A Systematic Review of Reproducibility Research in NLP**

Belz et al., EACL 2021.

This is probably the **first paper I'd read before designing your reproduction methodology**.

It systematically reviews reproducibility work in NLP and discusses different meanings of:

* reproducibility
* replicability
* repeatability
* experimental artifacts
* independent reproduction
* variation between implementations

It also points toward many individual reproduction studies that you can mine for additional references. ([ACL Anthology][7])

[Read the systematic review](https://aclanthology.org/2021.eacl-main.29/?utm_source=chatgpt.com)

---

### **The 2024 ReproNLP Shared Task on Reproducibility of Evaluations in NLP**

This is particularly interesting because it isn't just one paper doing one reproduction.

It is part of a sequence:

```text
ReproGen'21
     ↓
ReproGen'22
     ↓
ReproNLP'23
     ↓
ReproNLP'24
```

The shared tasks collect and compare **actual reproduction studies of NLP evaluations**. ([ACL Anthology][8])

That makes them a useful source for finding dozens of individual examples.

[ReproNLP 2024 overview](https://aclanthology.org/2024.humeval-1.9/?utm_source=chatgpt.com)

---

# 4. Particularly interesting recent LLM reproducibility work

There are now papers that move beyond classical NLP into **LLM evaluation reproducibility**.

### **A Looming Replication Crisis in Evaluating Behavior in Language Models? Evidence and Solutions**

Vaugrante, Niepert & Hagendorff.

This one is particularly relevant to your situation.

They attempted to replicate claims about prompting techniques across:

* GPT-3.5
* GPT-4o
* Gemini 1.5 Pro
* Claude 3 Opus
* Llama 3-8B
* Llama 3-70B

and found that many reported effects did **not survive their replication experiments**. They highlight methodological problems with how behavioral LLM experiments are evaluated. ([arXiv][9])

The important conceptual lesson is:

> **A reported performance difference isn't necessarily a robust property of the method.**

That is exactly the question you're facing with your Lizard modification.

---

## And there is a very recent 2026 paper that is almost tailor-made for your evaluation problem

### **ReproEvalCard: A Reporting Standard for Reproducible Evaluation of LLM Pipelines**

Pattnayak & Bhatia, ACL 2026.

They audited **55 LLM pipeline papers** from 2022–2025 and looked specifically at whether the artifacts necessary to reproduce the *evaluation* were available.

They found:

* **75%** were missing randomness controls
* **61%** were missing intermediate execution traces

They argue that model weights and datasets aren't enough: reproducible evaluation may also require things such as prompts, judge configuration, retrieval snapshots and intermediate traces. ([ACL Anthology][10])

[Read ReproEvalCard](https://aclanthology.org/2026.acl-short.22/?utm_source=chatgpt.com)

For your thesis, the analogous idea would be:

```text
Lizard reproduction package
│
├── exact git commit
├── model config
├── tokenizer
├── initialization
├── training data version
├── data ordering
├── optimizer
├── LR schedule
├── precision
├── checkpoint
├── evaluation version
├── MMLU prompts
├── few-shot examples
├── answer extraction
├── random seeds
└── intermediate checkpoints
```

---

# One paper I especially recommend for you: MultiBERTs

This is **probably more useful to your thesis than OpenLLaMA**.

The key sentence in MultiBERTs is essentially:

> A single pretrained model is an artifact; it does not necessarily tell you about the underlying training procedure.

They therefore trained **25 BERTs** with the same basic recipe but different random initialization and data order. They also released intermediate checkpoints for studying learning dynamics. ([arXiv][1])

This maps almost perfectly onto your question:

```text
                Lizard
                  │
        ┌─────────┴─────────┐
        │                   │
 Original Lizard        Your Lizard
        │                   │
   seed 1, 2, 3...     seed 1, 2, 3...
        │                   │
        └─────────┬─────────┘
                  ↓
          statistical comparison
```

Then:

```text
                 training
                    │
        ┌───────────┴───────────┐
        ↓                       ↓
Teacher / baseline          Your architecture
        │                       │
  hidden states            hidden states
  attention                GLA/SWA
  logits                   logits
  parameter changes        parameter changes
        │                       │
        └───────────┬───────────┘
                    ↓
              explanation
```

That would turn your current **"MMLU is 23% instead of 32%"** problem into a genuine reproducibility investigation.

---

# The papers I'd put in your thesis bibliography

If the goal is specifically to justify a **Lizard reproduction + discrepancy analysis + architectural modification**, I'd prioritize these:

| Priority | Paper                                                        | What it gives your thesis                                                            |
| -------: | ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
|       🥇 | **MultiBERTs**                                               | Multiple seeds, training variation, intermediate checkpoints, statistical robustness |
|       🥈 | **Reproducibility Issues for BERT-based Evaluation Metrics** | Concrete examples of preprocessing/evaluation causing reproduction failure           |
|       🥉 | **BERT4Rec Replicability Study**                             | Architecture/training implementation discrepancy investigation                       |
|        4 | **Replicability under Near-Perfect Conditions**              | Systematic investigation of missing implementation details                           |
|        5 | **Looming Replication Crisis in Evaluating Behavior in LMs** | Modern LLM replication methodology                                                   |
|        6 | **ReproEvalCard**                                            | Modern standard for documenting reproducible LLM evaluation                          |
|        7 | **Systematic Review of Reproducibility Research in NLP**     | Literature foundation and terminology                                                |
|        8 | **ReproNLP'24**                                              | Many concrete NLP reproduction cases                                                 |

![Image](https://images.openai.com/static-rsc-4/QEzUliff_sFUuuVbi7A0oqnD6WjTS--0Jg54QuT2VM__3iT6MJzv79_TINqUC62024KqNuldfPPXLTLhOX0SOSbHCQ4Ue9juO8CsJwRcD1O1eCttEPlcIQKoTuoN0Z-I4EWr22rmT2JudPfhi13MlXwWXASzraqkgY1geOL2XDUJMtWxY6b9m5Ha6VqW7tJo?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/XCdyIsACNZDvNr5kjMhnGeUqAiNy47uekgncoOa3vfgePNSqiV71hw0bRkqsLNemOmQ0NXrNVlGipFvAuoDsOyAXODlLisSUA_g5AxZeKKTipwn5OJB8esdsW_-mqqettboi3lcwR_2Xsq5Bl5SKxyippM2OoJaTMvDiv5NnIQZFGJRVSZULEU_KMMFtMbhR?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/MUtOs6IWdbfq0muCQr9B9HEJ2SjZBW6BUDk-oTc_Nhhxfyfu56lVHgW--AqZqQwRenb2VoZHI4w1xmzZ_NcOTtYJL-EW_bqDpzz4AudJwdqLbbxGgKmVg83mDJhTlp_hz4tIScuvCb_7XwKsvGYbNJYuS9HQJ8lmcS46sACMKX8Dyi2FAu07oysdnhanthBw?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/hzbbQHVE_jUgkIYEbkpiIOLQRMAJxEop3uKcKrmRo4ZQNnZR0yQEwQqdHRggzId94K3baRmkwRsANYqmFoXYlEN6AjYuzaPuFsf5rvRj3VIwUmaxTzjRJk3KcGouNkRvNeUHjYxKv_P-YZe9fHTOE8-KQVRQ8zSH5iG1YVWMWIRVhKLJ790VmKtjA_xk-ozZ?purpose=fullsize)

### Most important implication for your Lizard work

I would **not** make the thesis question simply:

> *"Why does my Lizard implementation achieve lower MMLU than the teacher?"*

I'd formulate the experimental problem more like:

> **"To what extent can the reported Lizard results be reproduced, what implementation and training factors explain deviations from the reported results, and how does the proposed GLA+SWA architecture affect these reproducibility and performance characteristics?"**

That framing gives you permission to report a **negative reproduction result** as a meaningful scientific result, provided you systematically isolate the causes.

And the literature above gives you a strong precedent for doing exactly that.

[1]: https://arxiv.org/abs/2106.16163?utm_source=chatgpt.com "The MultiBERTs: BERT Reproductions for Robustness Analysis"
[2]: https://aclanthology.org/2022.emnlp-main.192/?utm_source=chatgpt.com "Reproducibility Issues for BERT-based Evaluation Metrics - ACL Anthology"
[3]: https://arxiv.org/abs/2207.07483?utm_source=chatgpt.com "A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation"
[4]: https://aclanthology.org/2022.insights-1.23/?utm_source=chatgpt.com "Replicability under Near-Perfect Conditions – A Case-Study from Automatic Summarization - ACL Anthology"
[5]: https://research.google/pubs/the-multiberts-bert-reproductions-for-robustness-analysis/?utm_source=chatgpt.com "The MultiBERTs: BERT Reproductions for Robustness Analysis"
[6]: https://doi.org/10.1162/tacl_a_00349?utm_source=chatgpt.com "A Primer in BERTology: What We Know About How BERT Works | Transactions of the Association for Computational Linguistics | MIT Press"
[7]: https://aclanthology.org/2021.eacl-main.29/?utm_source=chatgpt.com "A Systematic Review of Reproducibility Research in Natural Language Processing - ACL Anthology"
[8]: https://aclanthology.org/2024.humeval-1.9/?utm_source=chatgpt.com "The 2024 ReproNLP Shared Task on Reproducibility of Evaluations in NLP: Overview and Results - ACL Anthology"
[9]: https://arxiv.org/abs/2409.20303?utm_source=chatgpt.com "A Looming Replication Crisis in Evaluating Behavior in Language Models? Evidence and Solutions"
[10]: https://aclanthology.org/2026.acl-short.22/?utm_source=chatgpt.com "ReproEvalCard: A Reporting Standard for Reproducible Evaluation of LLM Pipelines - ACL Anthology"
