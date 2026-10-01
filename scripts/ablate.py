"""
Evaluate Lizard with one branch switched off at inference time (no retraining)
-> ABLATE=no_awa: y = (1 + alpha) * GLA   (sliding window branch off)
-> ABLATE=no_gla: y = (1 + alpha) * AWA   (gated linear branch off)
Each branch alone is a weighted average of the values, so scaling by (1 + alpha)
keeps the output on the scale of GLA + alpha * AWA that the next layer expects
-> Run from the lolcats repo root, with eval_lm_harness.py's arguments:
   ABLATE=no_gla PYTHONPATH=. python scripts/ablate.py --model_type lolcats_ckpt ...
"""
import os
import runpy
import sys

import src.model.linear_attention.lizard_attention as lz

MODE = os.environ.get('ABLATE')
assert MODE in ('no_awa', 'no_gla'), 'Set ABLATE=no_awa or ABLATE=no_gla'


def lizard(self, q, k, v, gamma):
    dtype = q.dtype
    q, k, v, alpha, meta = lz.upcast(q, k, v, self.alpha_blend, self.meta_tokens)
    if MODE == 'no_awa':
        fq, fk = self.feature_maps(q, k)
        y = (1 + alpha) * lz.gla(fq, fk, v, gamma)
    else:
        y = (1 + alpha) * lz.awa(q, k, v, meta, self.window)
    return y.to(dtype)


lz.LolcatsLizardAttention.lizard = lizard
# lm-eval scores each request with one forward pass on a fresh cache, so skip the cache
# (fine for loglikelihood tasks like MMLU and PIQA; generation would need the recurrent path)
lz.LolcatsLizardAttention.lizard_recurrent = lambda self, q, k, v, gamma, cache: self.lizard(q, k, v, gamma)
print(f'-> Lizard ablation: {MODE}')

if __name__ == '__main__':
    sys.argv = ['lm_eval_harness/eval_lm_harness.py'] + sys.argv[1:]
    runpy.run_path('lm_eval_harness/eval_lm_harness.py', run_name='__main__')
