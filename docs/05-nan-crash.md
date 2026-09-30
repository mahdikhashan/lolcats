# 5. NaN crash in stage 2

## What happened

In the first full run (W&B `zyt5syvy`), stage 2 crashed in epoch 0 at iteration ~3,094 of 4,714,
gradient step 386, learning rate 1e-4, sequence length 2048. Until then the loss and perplexity
looked healthy (perplexity about 12–21). The end of the log:

```
UserWarning: Error detected in LogSoftmaxBackward0. No forward pass information available.
Enable detect anomaly during forward pass...
> /workspace/lolcats/src/trainer/default_lm.py(174)train_step()
-> if (self.step + 1) % accum_iter == 0:  # and self.step != 0:
...
bdb.BdbQuit
make: *** [Makefile:42: lizard] Error 1
```

Two separate things went wrong:

1. **A NaN appeared.** Anomaly detection flags the first backward function whose output contains
   NaN. `LogSoftmaxBackward0` belongs to the cross-entropy loss, so the logits were most likely
   already NaN in the forward pass.
2. **The debugger stopped the job.** `default_lm.py` wraps `loss.backward()` like this:

   ```python
   try:
       with torch.autograd.set_detect_anomaly(True):
           loss.backward()
   except Exception as e:
       breakpoint()
   ```

   In a non-interactive job, `pdb` exits with `bdb.BdbQuit`, which killed the process. It was the
   breakpoint, not the NaN itself, that ended the run.

## Cause: two unguarded operations in the attention

Reading `lizard_attention.py` for operations that can produce NaN in float32:

- **Window branch (AWA): `exp` without subtracting the maximum.** It computed
  `exp(q·k / sqrt(d))` directly. In float32 `exp` overflows to `inf` above about 88.7, and
  `inf / inf = NaN`. q·k grows as LoRA trains the q/k projections in stage 2, which fits a crash
  that appears suddenly, deep into stage 2.
- **Gated branch (GLA): possible 0 / 0.** The output is `(w @ v) / w.sum(-1)`. If every weight in a
  row underflows to 0, this is 0 / 0.

The Hedgehog feature maps use `softmax`, which already subtracts the maximum, and `sigmoid` and the
cumulative product of gates only underflow to 0, which is harmless.

The overflow is the most likely cause, but it was not reproduced on the real checkpoint.

## Fix (PR #5, commit `cb8bc80`)

Only `src/model/linear_attention/lizard_attention.py` changed:

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

The recurrent decode path got the same two changes. The shift by the row maximum (over the scores
and the sink logits) cancels exactly in the ratio, so the result is mathematically unchanged. The
maximum is detached, which is exact because softmax is shift-invariant.

### Verification

No GPU or matching PyTorch was available at the time, so the formulas were re-implemented in NumPy
(float32, with scores kept in float32):

| Check | Result |
|---|---|
| Normal scores: old vs new formula | max difference 3.6e-7 |
| Scores scaled so the maximum reaches ~156 (> 88.7): old formula | NaN |
| Same scores: new formula | finite; matches float64 to 1.4e-5 |
| One decode step vs the last row of the full computation | match |

Later, the full reference test suite from jku-thesis passed on the fixed code, 26 of 26
([document 8](08-verification.md)).

## What was deliberately left out

The first version of the fix also changed the trainer. At the user's request the PR was reduced to
the attention fix only, one change per PR. Removed from PR #5:

- removing the anomaly detection and the `breakpoint()` (the breakpoint is still there, so any
  future NaN would still stop a job the same way);
- skipping batches with a non-finite loss and gradient steps with a non-finite gradient norm;
- optional gradient clipping (`max_grad_norm`, set to 1.0 for finetuning);
- the W&B step fix, so stage 2 metrics still don't reach W&B.

Gradient clipping at 1.0 is part of the Lizard paper's recipe (Table 13), so it may come back as its
own change (see [document 10](10-open-issues-and-next-steps.md)).

## Recovery

Stage 1 had finished and its checkpoint was fine: the fix changes how the math is computed, not the
parameters. So only stage 2 was rerun, from that checkpoint, with `make hf-job-finetune` (PR #6),
after rebuilding the image from `main` with the fix ([document 4](04-training-runs.md), Run 2).
The rerun finished without a NaN.
