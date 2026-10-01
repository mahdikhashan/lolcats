# 5. NaN crash in stage 2

## Events

In Run 1 (W&B `zyt5syvy`), stage 2 crashed in epoch 0, at iteration ~3,094 of 4,714. The crash occurred at gradient step 386, with learning rate 1e-4 and sequence length 2048. Until then, the loss and the perplexity looked normal (perplexity approximately 12–21). The end of the log:

```
UserWarning: Error detected in LogSoftmaxBackward0. No forward pass information available.
Enable detect anomaly during forward pass...
> /workspace/lolcats/src/trainer/default_lm.py(174)train_step()
-> if (self.step + 1) % accum_iter == 0:  # and self.step != 0:
...
bdb.BdbQuit
make: *** [Makefile:42: lizard] Error 1
```

Two separate problems occurred:

1. **A NaN appeared.** Anomaly detection reports the first backward function with NaN in its output. `LogSoftmaxBackward0` is part of the cross-entropy loss. Thus the logits were most probably NaN already in the forward pass.
2. **The debugger stopped the job.** `default_lm.py` contains this code around `loss.backward()`:

   ```python
   try:
       with torch.autograd.set_detect_anomaly(True):
           loss.backward()
   except Exception as e:
       breakpoint()
   ```

   In a job without an interactive terminal, `pdb` exits with `bdb.BdbQuit`, and this exit stopped the process. The breakpoint ended the run, not the NaN itself.

## Cause: two operations in the attention without protection

A review of `lizard_attention.py` looked for operations that can give NaN in float32. It found two:

- **Window branch (AWA): `exp` without subtraction of the maximum.** The code calculated `exp(q·k / sqrt(d))` directly. In float32, `exp` overflows to `inf` above approximately 88.7, and `inf / inf = NaN`. In stage 2, LoRA trains the q/k projections, and q·k increases. This agrees with a crash that occurs suddenly, late in stage 2.
- **Gated branch (GLA): possible 0 / 0.** The output is `(w @ v) / w.sum(-1)`. If every weight in a row underflows to 0, the result is 0 / 0.

The other operations are safe:

- The Hedgehog feature maps use `softmax`, which subtracts the maximum already.
- `sigmoid` and the cumulative product of gates can only underflow to 0, and this causes no problem.

The overflow is the most probable cause. But no test reproduced the overflow on the real checkpoint.

## Fix (PR #5, commit `cb8bc80`)

The fix changed only `src/model/linear_attention/lizard_attention.py`:

```python
def gla(fq, fk, v, gamma):
    w = (fq @ fk.transpose(-1, -2)) * gate_products(gamma)[:, None]
    # Guard 0 / 0 when every weight in a row underflows
    return (w @ v) / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)


def sink_softmax(scores, meta):
    """
    exp(scores) / (sum(exp(meta)) + sum(exp(scores))) over the last dim, shifted
    by the row max over scores and meta so exp cannot overflow
    """
    m = torch.maximum(scores.amax(-1, keepdim=True), meta.max()).detach()
    e = (scores - m).exp()
    return e / ((meta - m).exp().sum(-1, keepdim=True) + e.sum(-1, keepdim=True))


def awa(q, k, v, meta, window):
    mask = window_mask(q.shape[-2], window, q.device)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return sink_softmax(scores.masked_fill(~mask, float("-inf")), meta) @ v
```

The recurrent decode path received the same two changes. The code subtracts the maximum of each row, over the scores and the sink logits. This shift cancels exactly in the ratio, so the mathematical result does not change. The code detaches the maximum from the gradient. This is exact, because the softmax result does not change when all its inputs move by the same value.

### Checks

At that time, no GPU and no matching PyTorch version were available. Thus the checks used a NumPy version of the formulas, in float32, with the scores also in float32.

| Check | Result |
|---|---|
| Normal scores: old formula against new formula | Maximum difference 3.6e-7 |
| Scores scaled to a maximum of ~156 (> 88.7): old formula | NaN |
| The same scaled scores: new formula | Finite. Matches float64 to 1.4e-5. |
| One decode step against the last row of the full calculation | Match |

Later, the reference test suite from jku-thesis ran on the code with the fix. All 26 selected tests passed. The run deselected 2 tests ([document 8](08-verification.md)).

## Parts that PR #5 does not include

The first version of the fix also changed the trainer. The user asked for one change per PR. Thus the PR kept only the attention fix. These changes are not in PR #5:

- The removal of the anomaly detection and the `breakpoint()`. The breakpoint stays in the code. Thus a NaN in the future will stop a job in the same way.
- Skipping batches with a non-finite loss, and skipping gradient steps with a non-finite gradient norm.
- Optional gradient clipping (`max_grad_norm`, set to 1.0 for finetuning).
- The fix of the W&B step counter. Thus the stage 2 metrics still do not go to W&B.

Gradient clipping at 1.0 is part of the recipe of the paper (Table 13). Thus it may return as a separate change ([document 10](10-open-issues-and-next-steps.md)).

## Recovery

Stage 1 finished before the crash, and its checkpoint had no problem. The fix changes how the code calculates the equations, not the parameters. The recovery had three steps:

1. A new build of the image from `main`, with the fix.
2. A new run of stage 2 only, from the stage 1 checkpoint, with `make hf-job-finetune` (PR #6). [Document 4](04-training-runs.md) calls this run Run 2.
3. Run 2 finished with no NaN.
