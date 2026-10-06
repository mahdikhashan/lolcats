# Appendix: definitions

Source: an answer from ChatGPT (2026-10-06). This appendix keeps the answer without changes and does not check it.

---

Here are the main concepts in simple terms:

| Concept                            | Simple explanation                                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Fine-tuning**                    | Further training an already pretrained model on a specific dataset.                                                                   |
| **Causal Language Modeling (CLM)** | Training the model to **predict the next token** given the previous tokens. This is the same basic objective used during pretraining. |
| **RLHF**                           | **Reinforcement Learning from Human Feedback** — training the model to produce outputs humans prefer.                                 |
| **Packing**                        | A way to **reduce wasted space** when training on texts of different lengths.                                                         |
| **EOS token**                      | **End-of-Sequence** token. It marks where one text ends and another begins.                                                           |
| **Padding**                        | Adding meaningless extra tokens so all sequences in a batch have the same length. This wastes computation.                            |
| **Context size**                   | The maximum number of tokens the model can process at once, e.g. 4K or 8K tokens.                                                     |
| **Batch**                          | A group of training examples processed together.                                                                                      |

### Packing in one example

Without packing:

```text
Text A: [tokens........] [PAD] [PAD]
Text B: [tokens.....]    [PAD] [PAD]
```

With packing:

```text
[Text A] [EOS] [Text B] [EOS] [Text C] [EOS] ...
```

Then they cut this long sequence into **fixed-size chunks** matching the model's context size.

**Main idea:** packing keeps the GPU busy with real tokens instead of wasting computation on padding.
