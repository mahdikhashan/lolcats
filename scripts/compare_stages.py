"""
Find where the gap starts: the teacher, Lizard after stage 1 and Lizard after stage 2
(docs/11-gap-analysis.md, section 10). scripts/compare_stages.sh runs both steps below, from the repo root.
-> eval: run lm_eval_harness/eval_lm_harness.py for one model and task, and also write to OUT_DIR
   PYTHONPATH=. python scripts/compare_stages.py eval OUT_DIR <eval_lm_harness.py arguments>
   - results.json: the harness results
   - <task>_write_out_info.json: per question, the log-likelihood of each choice, the right answer and acc
   - checkpoints.json: per checkpoint, its SHA-256, stored step and loss, dtypes, and whether every trainable
     parameter of the model holds the checkpoint value (sections 3 and 9)
   - lizard.json: per layer, the gate values on one 5-shot MMLU prompt, alpha, the sink logits and the
     feature-map weight sizes (section 6, and factors 0 and 1 of section 12)
-> summary: compare the models in RUN_DIR/<model>/<task>/, read the training results CSVs in
   RUN_DIR/training/ (if any), and write RUN_DIR/summary.md and summary.json
   python scripts/compare_stages.py summary RUN_DIR
-> TEMPERATURE=T (default 1) divides the logits by T before the harness takes log_softmax
   (docs/experiments/temperature.md). scripts/temperature.sh runs one RUN_DIR per temperature in ROOT/T=<t>/;
   temperatures compares them and writes ROOT/summary.md and summary.json
   python scripts/compare_stages.py temperatures ROOT
"""
import csv
import hashlib
import json
import math
import os
import random
import runpy
import sys
from collections import Counter
from os.path import isdir, isfile, join

import torch

MODELS = {
    'teacher': 'A. Teacher',
    'stage1': 'B. Lizard after stage 1 (no LoRA)',
    'stage2': 'C. Lizard after stage 2',
}
TASKS = {
    'mmlu_subset': 'MMLU subset (5-shot, 5 questions per subject)',
    'mmlu': 'MMLU, all questions (5-shot)',
    'piqa': 'PIQA (0-shot)',
    'arc_easy': 'ARC-Easy (0-shot)',
}
LIZARD_PARAMS = ('phi_q.weight', 'phi_k.weight', 'W_gamma.weight', 'W_gamma.bias', 'meta_tokens', 'alpha_blend')
GATE_TASK = os.environ.get('GATE_TASK', 'hendrycksTest-high_school_us_history')  # long reading passages
GATE_MAX_TOKENS = 2048
KEPT_AFTER = (128, 256, 512)  # tokens back from the last token of the prompt
LETTERS = 'ABCD'
TEMPERATURE = float(os.environ.get('TEMPERATURE', '1'))


def dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)


def load(path):
    with open(path) as f:
        return json.load(f)


def layer_index(name):
    return int(name.split('layers.')[1].split('.')[0])


# --- eval ---

def check_checkpoint(model, path, stage):
    """
    Metadata of a checkpoint, and whether the model's trainable parameters hold its values
    -> The loaders only assert that the checkpoint has no unexpected keys (gap analysis, section 3)
    -> A stage 1 checkpoint must hold every Lizard parameter, a stage 2 checkpoint every LoRA weight
    """
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    with open(path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    # LoRA wraps the model in base_model.model, which stage 1 names don't have
    strip = lambda n: n[len('base_model.model.'):] if n.startswith('base_model.model.') else n
    params = {strip(n): p for n, p in model.named_parameters()}
    tensors = {strip(n): t for n, t in state_dict.items()}
    if stage == 'stage1':
        expected = {n for n in params if n.endswith(LIZARD_PARAMS)}
    else:
        expected = {n for n in params if '.lora_' in n}
    not_loaded = sorted(n for n in expected & set(tensors)
                        if not torch.equal(params[n].detach().cpu(), tensors[n].to(params[n].dtype)))
    n_params = sum(t.numel() for t in state_dict.values())
    return {
        'path': path,
        'sha256': sha256,
        'bytes': os.path.getsize(path),
        'step': checkpoint.get('step'),
        'losses': {k: float(v) for k, v in checkpoint.items() if 'loss' in k},
        'tensors': len(state_dict),
        'parameters': n_params,
        'bytes_per_parameter': os.path.getsize(path) / max(n_params, 1),
        'dtypes': dict(Counter(str(t.dtype) for t in state_dict.values())),
        'expected_keys': len(expected),
        'missing_keys': sorted(expected - set(tensors)),
        'unexpected_keys': sorted(set(tensors) - set(params)),
        'values_not_loaded': not_loaded,
    }


def lizard_stats(model, tokenizer):
    """
    Per layer: gate values on one 5-shot MMLU prompt, and the trained Lizard parameters
    -> At initialization, W_gamma = 0 (so gamma = 0.5), alpha = 1 and the feature-map weights have RMS 0.02
    """
    from lm_eval.tasks import get_task_dict
    task = get_task_dict([GATE_TASK])[GATE_TASK]
    doc = next(iter(task.test_docs()))
    prompt = task.fewshot_context(doc=doc, num_fewshot=5, rnd=random.Random(42))
    # As the harness does for these models: no beginning-of-text token, keep the end of long prompts
    ids = tokenizer(prompt, add_special_tokens=False).input_ids[-GATE_MAX_TOKENS:]
    ids = torch.tensor([ids], device=next(model.parameters()).device)

    gammas = {}
    hooks = [module.register_forward_hook(
                 lambda m, i, o, layer=layer_index(name):
                 gammas.__setitem__(layer, torch.sigmoid(o.float())[0, :, 0].double().cpu()))
             for name, module in model.named_modules() if name.endswith('W_gamma')]
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    layers = []
    for name, attn in model.named_modules():
        if not (name.endswith('self_attn') and hasattr(attn, 'alpha_blend')):
            continue
        gamma = gammas[layer_index(name)]
        phi_q, phi_k = attn.phi_q.weight.float(), attn.phi_k.weight.float()
        layers.append({
            'layer': layer_index(name),
            'gamma_mean': gamma.mean().item(),
            'gamma_std': gamma.std().item(),
            'gamma_min': gamma.min().item(),
            'gamma_max': gamma.max().item(),
            'gamma_below_1e-3': (gamma < 1e-3).double().mean().item(),
            'gamma_above_0.999': (gamma > 0.999).double().mean().item(),
            # Weight the gated branch keeps on a token w positions before the last token
            'kept_after': {str(w): gamma[-w:].log().sum().exp().item() if len(gamma) >= w else None
                           for w in KEPT_AFTER},
            'alpha': attn.alpha_blend.float().mean().item(),  # the mean over the heads for lizard_v2 alpha_per_head
            'alpha_per_head': attn.alpha_blend.float().tolist() if attn.alpha_blend.dim() else None,
            'sink_logits': attn.meta_tokens.float().tolist(),
            'W_gamma_norm': attn.W_gamma.weight.float().norm().item(),
            'phi_q_rms': phi_q.pow(2).mean().sqrt().item(),
            'phi_q_max_abs': phi_q.abs().max().item(),
            'phi_k_rms': phi_k.pow(2).mean().sqrt().item(),
            'phi_k_max_abs': phi_k.abs().max().item(),
        })
    return {'gate_task': GATE_TASK, 'prompt_tokens': ids.shape[1],
            'layers': sorted(layers, key=lambda l: l['layer'])}


def run_eval(out_dir, harness_args):
    sys.path.append(os.environ.get('LM_EVALUATION_HARNESS_PATH', '/workspace/lm-evaluation-harness'))
    from lm_eval import evaluator
    import src.model.load_model_for_eval as loader
    os.makedirs(out_dir, exist_ok=True)

    if TEMPERATURE != 1:
        assert TEMPERATURE > 0, f'TEMPERATURE must be positive, got {TEMPERATURE}'
        # The harness takes log_softmax of what _model_call returns, for every causal model
        from lm_eval.models.huggingface import AutoCausalLM
        model_call = AutoCausalLM._model_call

        def scaled_model_call(self, inputs, labels=None):
            return model_call(self, inputs, labels) / TEMPERATURE

        AutoCausalLM._model_call = scaled_model_call
        print(f'-> Temperature {TEMPERATURE}: logits divided by it before log_softmax')

    simple_evaluate = evaluator.simple_evaluate

    def logged_evaluate(**kwargs):
        # Per-question outputs give paired statistics and the answer letters
        results = simple_evaluate(**{**kwargs, 'write_out': True, 'output_base_path': out_dir})
        results['temperature'] = TEMPERATURE
        # With one token per answer letter, a temperature cannot change the MMLU prediction
        results['answer_letter_tokens'] = {l: len(kwargs['model'].tok_encode(f' {l}')) for l in LETTERS}
        dump(results, join(out_dir, 'results.json'))
        return results

    load_model_from_checkpoint = loader.load_model_from_checkpoint

    def logged_load(**kwargs):
        lm, model_config, tokenizer = load_model_from_checkpoint(**kwargs)
        model = getattr(lm, 'model', lm)
        checkpoints = {}
        for stage, key in (('stage1', 'attn_mlp_checkpoint_path'), ('stage2', 'finetune_checkpoint_path')):
            if kwargs.get(key) is not None:
                checkpoints[stage] = check_checkpoint(model, kwargs[key], stage)
                c = checkpoints[stage]
                print(f'-> {stage} checkpoint: {c["expected_keys"]} expected trainable keys, '
                      f'{len(c["missing_keys"])} missing, {len(c["unexpected_keys"])} unexpected, '
                      f'{len(c["values_not_loaded"])} not loaded, step {c["step"]}, losses {c["losses"]}')
        dump(checkpoints, join(out_dir, 'checkpoints.json'))
        dump(lizard_stats(model, tokenizer), join(out_dir, 'lizard.json'))
        return lm, model_config, tokenizer

    evaluator.simple_evaluate = logged_evaluate
    loader.load_model_from_checkpoint = logged_load
    sys.argv = ['lm_eval_harness/eval_lm_harness.py'] + harness_args
    runpy.run_path('lm_eval_harness/eval_lm_harness.py', run_name='__main__')


# --- summary ---

def choice_stats(logits):
    """Probability mass on the choices, and the confidence and entropy (bits) over the choices"""
    m = max(logits)
    p = [math.exp(x - m) for x in logits]
    p = [x / sum(p) for x in p]
    return {'mass': sum(math.exp(x) for x in logits), 'confidence': max(p),
            'entropy_bits': -sum(x * math.log2(x) for x in p if x > 0)}


def read_run(run_dir):
    """Score, standard error and per-question results of one model on one task"""
    harness = load(join(run_dir, 'results.json'))
    results = harness['results']
    questions = {}
    for f in sorted(os.listdir(run_dir)):
        if f.endswith('_write_out_info.json'):
            task = f[:-len('_write_out_info.json')]
            for d in load(join(run_dir, f)):
                logits = [d[k] for k in sorted((k for k in d if k.startswith('logit_')),
                                               key=lambda k: int(k.split('_')[1]))]
                questions[(task, d['doc_id'])] = {
                    'acc': float(d['acc']),
                    'acc_norm': float(d['acc_norm']) if 'acc_norm' in d else None,
                    'prediction': max(range(len(logits)), key=lambda i: logits[i]),
                    'truth': d['truth'],
                    'logits': logits,
                    **choice_stats(logits),
                }
    run = {'n': len(questions), 'questions': questions, 'temperature': harness.get('temperature', 1.0),
           'answer_letter_tokens': harness.get('answer_letter_tokens')}
    for stat in ('mass', 'confidence', 'entropy_bits'):
        run[f'mean_{stat}'] = sum(q[stat] for q in questions.values()) / max(len(questions), 1)
    subjects = [r['acc'] for t, r in results.items() if t.startswith('hendrycksTest-')]
    if subjects:  # The harness MMLU score: unweighted mean over subjects, binomial SE as in docs/07
        run['mmlu'] = True
        run['acc'] = sum(subjects) / len(subjects)
        run['acc_stderr'] = math.sqrt(run['acc'] * (1 - run['acc']) / run['n'])
    else:
        (r,) = results.values()
        run['mmlu'] = False
        for metric in ('acc', 'acc_stderr', 'acc_norm', 'acc_norm_stderr'):
            run[metric] = r.get(metric)
    for name in ('checkpoints.json', 'lizard.json'):
        if isfile(join(run_dir, name)):
            run[name[:-len('.json')]] = load(join(run_dir, name))
    return run


def read_training(run_dir):
    """First, best and last validation rows of each training results CSV (gap analysis, section 9)"""
    found = {}
    folder = join(run_dir, 'training')
    for f in sorted(os.listdir(folder)) if isdir(folder) else []:
        if not f.endswith('.csv'):
            continue
        with open(join(folder, f)) as fh:  # the first column is the pandas index, with an empty name
            rows = [{k: float(v) for k, v in row.items() if k and v} for row in csv.DictReader(fh)]
        metric = next((k for k in rows[0] if 'loss' in k), None) if rows else None
        if metric is None:
            continue
        found['stage1' if f.endswith('_distill.csv') else 'stage2'] = {
            'file': f, 'metric': metric, 'evaluations': len(rows),
            'first': rows[0], 'best': min(rows, key=lambda r: r[metric]), 'last': rows[-1]}
    return found


def paired(a, b):
    """Accuracy of a minus accuracy of b on their common questions, with the paired SE, in points"""
    common = sorted(set(a['questions']) & set(b['questions']))
    d = [a['questions'][q]['acc'] - b['questions'][q]['acc'] for q in common]
    n = len(d)
    if n < 2:
        return None
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n)
    return {'n': n, 'diff': 100 * mean, 'se': 100 * se, 'z': mean / se if se > 0 else None}


def outcome(stage1_drop, stage2_drop, total_drop):
    """The reading of section 10. A drop or a gain is clear when it is more than 2 paired SE"""
    clear = lambda p: p['diff'] > 2 * p['se']
    gain = lambda p: -p['diff'] > 2 * p['se']
    if clear(stage1_drop) and not clear(stage2_drop):
        text = 'Stage 1 failure: the attention approximation is the root problem'
        return text + (', and stage 2 recovers part of it' if gain(stage2_drop) else '')
    if clear(stage2_drop) and not clear(stage1_drop):
        return 'Stage 2 failure: the root is the finetuning'
    if clear(stage1_drop) and clear(stage2_drop):
        return 'Both stages add a clear drop'
    if clear(total_drop):
        return 'No clear drop at one stage, but a clear drop from the teacher to stage 2'
    return 'Good result: no clear drop. Stage 2 has no problem'


def fmt(x, digits=1, scale=100):
    return '–' if x is None else f'{scale * x:.{digits}f}'


def fmt_tokens(tokens):
    """Tokens per answer letter: one number when all letters agree"""
    if not tokens:
        return '–'
    return str(next(iter(set(tokens.values())))) if len(set(tokens.values())) == 1 else str(tokens)


def summarize(run_dir):
    runs = {m: {t: read_run(join(run_dir, m, t)) for t in TASKS if isfile(join(run_dir, m, t, 'results.json'))}
            for m in MODELS if isdir(join(run_dir, m))}
    models = [m for m in MODELS if runs.get(m)]
    tasks = [t for t in TASKS if any(t in runs[m] for m in models)]
    out = ['# Stage comparison: where the gap starts', '',
           f'This run tests section 10 of `docs/11-gap-analysis.md`. Run directory: `{run_dir}`.', '']
    if isfile(join(run_dir, 'env.txt')):
        out += ['```', open(join(run_dir, 'env.txt')).read().strip(), '```', '']

    # Scores
    out += ['## Scores', '', 'Accuracy in %, ± the binomial SE. n is the number of questions.', '',
            '| Task | ' + ' | '.join(MODELS[m] for m in models) + ' |', '|---' * (len(models) + 1) + '|']
    for t in tasks:
        cell = lambda r, k: f"{fmt(r[k])} ± {fmt(r[k + '_stderr'])} (n = {r['n']})" if r.get(k) is not None else '–'
        out.append(f'| {TASKS[t]} | ' + ' | '.join(cell(runs[m][t], 'acc') if t in runs[m] else '–'
                                                    for m in models) + ' |')
        if any(t in runs[m] and runs[m][t].get('acc_norm') is not None for m in models):
            out.append(f'| {TASKS[t]}, normalized | ' + ' | '.join(
                cell(runs[m][t], 'acc_norm') if t in runs[m] else '–' for m in models) + ' |')
    out.append('')

    # Where the drop happens
    comparisons = {}
    if all(m in models for m in MODELS):
        out += ['## Where the drop happens', '',
                'Accuracy differences in points (first model minus second model), with the paired SE over the '
                'same questions. A positive value is a drop. A drop is clear when it is more than 2 SE. Each '
                'question counts equally, which matches the subject mean when every subject has the same number '
                'of questions, as in the MMLU subset.', '',
                '| Task | Teacher → stage 1 | Stage 1 → stage 2 | Teacher → stage 2 | Outcome |',
                '|---|---|---|---|---|']
        for t in tasks:
            if not all(t in runs[m] for m in MODELS):
                continue
            c = {'teacher_to_stage1': paired(runs['teacher'][t], runs['stage1'][t]),
                 'stage1_to_stage2': paired(runs['stage1'][t], runs['stage2'][t]),
                 'teacher_to_stage2': paired(runs['teacher'][t], runs['stage2'][t])}
            if None in c.values():
                continue
            c['outcome'] = outcome(c['teacher_to_stage1'], c['stage1_to_stage2'], c['teacher_to_stage2'])
            comparisons[t] = c
            cell = lambda p: f"{p['diff']:+.1f} ± {p['se']:.1f} (z {p['z']:.1f})" if p['z'] is not None \
                else f"{p['diff']:+.1f} ± {p['se']:.1f}"
            out.append(f"| {TASKS[t]} | {cell(c['teacher_to_stage1'])} | {cell(c['stage1_to_stage2'])} | "
                       f"{cell(c['teacher_to_stage2'])} | {c['outcome']} |")
        out += ['', 'Reading of section 10 (example values of MMLU-subset accuracy):', '',
                '| Outcome | Teacher | After stage 1 | After stage 2 | Conclusion |',
                '|---|---|---|---|---|',
                '| Good result | 33.7 | 30–33 | 30–33 | Stage 2 has no problem. |',
                '| Stage 1 failure | 33.7 | 24 | 23 | The attention approximation is the root problem. |',
                '| Stage 2 failure | 33.7 | 31 | 23 | The root is clearly the finetuning. |', '']

    # Answer letters on MMLU (section 4)
    letters = {}
    for t in tasks:
        mmlu_models = [m for m in models if t in runs[m] and runs[m][t]['mmlu']]
        if not mmlu_models:
            continue
        out += [f'## Answer letters: {TASKS[t]}', '',
                'Share of each predicted letter, in %. On MMLU, the Lizard model of Run 2 selected "A" for almost '
                'every question (section 4 of the gap analysis).', '',
                '| Model | A | B | C | D | Accuracy | Accuracy if always "A" |', '|---|---|---|---|---|---|---|']
        for m in mmlu_models:
            qs = runs[m][t]['questions'].values()
            predicted = Counter(q['prediction'] for q in qs)
            shares = [predicted[i] / len(qs) for i in range(4)]
            always_a = sum(q['truth'] == 0 for q in qs) / len(qs)
            letters[f'{m}/{t}'] = {'predicted': dict(zip(LETTERS, shares)), 'always_a_accuracy': always_a}
            out.append(f'| {MODELS[m]} | ' + ' | '.join(fmt(s) for s in shares)
                       + f" | {fmt(sum(q['acc'] for q in qs) / len(qs))} | {fmt(always_a)} |")
        truth = Counter(q['truth'] for q in runs[mmlu_models[0]][t]['questions'].values())
        n = sum(truth.values())
        out += ['', 'Right answers: ' + ', '.join(f'{l} {fmt(truth[i] / n)}%' for i, l in enumerate(LETTERS)), '',
                'Choice probabilities, as means over the questions. Mass: the probability of " A" to " D" '
                'together. Confidence: the largest of the four probabilities after normalization over the '
                'four letters. Entropy: over the four letters, in bits (maximum 2).', '',
                '| Model | Temperature | Tokens per letter | Mass | Confidence | Entropy (bits) |',
                '|---|---|---|---|---|---|']
        for m in mmlu_models:
            r = runs[m][t]
            out.append(f"| {MODELS[m]} | {r['temperature']:g} | {fmt_tokens(r['answer_letter_tokens'])} | "
                       f"{r['mean_mass']:.3f} | {r['mean_confidence']:.3f} | {r['mean_entropy_bits']:.3f} |")
        out.append('')

    # Checkpoints (sections 3 and 9)
    checks = {f'{m}/{t}': runs[m][t]['checkpoints'] for m in models for t in runs[m] if 'checkpoints' in runs[m][t]}
    if checks:
        files = {}
        for c in checks.values():
            for stage, info in c.items():
                files.setdefault(info['sha256'], (stage, info))
        out += ['## Checkpoints', '', 'Sections 3 and 9 of the gap analysis.', '',
                '| Stage | File | SHA-256 | Size | Parameters | Dtypes | Bytes per parameter | Stored step | Stored losses |',
                '|---|---|---|---|---|---|---|---|---|']
        for stage, info in files.values():
            out.append(f"| {stage} | `{os.path.basename(info['path'])}` | `{info['sha256']}` | {info['bytes']:,} B | "
                       f"{info['parameters']:,} | {info['dtypes']} | {info['bytes_per_parameter']:.2f} | "
                       f"{info['step']} | {info['losses']} |")
        out += ['', 'Trainable keys: for each evaluation, each expected trainable parameter of the model must be in '
                'the checkpoint and hold its value after the load.', '',
                '| Evaluation | Checkpoint | Expected | Missing | Unexpected | Not loaded |', '|---|---|---|---|---|---|']
        problems = False
        for run, c in checks.items():
            for stage, info in c.items():
                bad = info['missing_keys'] + info['unexpected_keys'] + info['values_not_loaded']
                problems |= bool(bad) or info['expected_keys'] == 0
                out.append(f"| {run} | {stage} | {info['expected_keys']} | {len(info['missing_keys'])} | "
                           f"{len(info['unexpected_keys'])} | {len(info['values_not_loaded'])} |")
        out += ['', 'Result: ' + ('**problem: see the lists in checkpoints.json**' if problems else
                                  'every expected trainable parameter loaded from its checkpoint.'), '']

    # Validation loss during training (section 9)
    training = read_training(run_dir)
    if training:
        stored = {stage: info['step'] for c in checks.values() for stage, info in c.items()}
        out += ['## Validation loss during training', '',
                'From the training results CSVs in `training/`. The trainer saves the checkpoint at the best step, '
                'so the stored step of the checkpoint should equal the best step.', '',
                '| Stage | File | Evaluations | First: step, loss | Best: step, loss | Last: step, loss | '
                'Stored step of the checkpoint |', '|---|---|---|---|---|---|---|']
        for stage, t in training.items():
            point = lambda r: f"{r.get('eval_step', float('nan')):.0f}, {r[t['metric']]:.4f}"
            out.append(f"| {stage} | `{t['file']}` | {t['evaluations']} | {point(t['first'])} | "
                       f"{point(t['best'])} | {point(t['last'])} | {stored.get(stage, '–')} |")
        out.append('')

    # Lizard parameters and gates (section 6, factors 0 and 1 of section 12)
    stats = {}
    for m in models:
        found = [runs[m][t]['lizard'] for t in runs[m] if 'lizard' in runs[m][t]]
        if not found:
            continue
        s = stats[m] = found[0]
        out += [f"## Lizard parameters and gates: {MODELS[m]}", '',
                f"Gate values on one 5-shot prompt of `{s['gate_task']}` ({s['prompt_tokens']} tokens). "
                'Kept after w: the weight that the gated branch keeps on a token w positions before the last token. '
                'Initial values: gamma = 0.5, alpha = 1, feature-map weight RMS 0.02.', '',
                '| Layer | Gamma mean | Std | Min | Max | < 1e-3 | > 0.999 | '
                + ' | '.join(f'Kept after {w}' for w in KEPT_AFTER)
                + ' | Alpha | Sink logits | φq RMS / max | φk RMS / max | ‖W_γ‖ |',
                '|---' * (12 + len(KEPT_AFTER)) + '|']
        for l in s['layers']:
            kept = [l['kept_after'][str(w)] for w in KEPT_AFTER]
            out.append(f"| {l['layer']} | {l['gamma_mean']:.3f} | {l['gamma_std']:.3f} | {l['gamma_min']:.3f} | "
                       f"{l['gamma_max']:.3f} | {fmt(l['gamma_below_1e-3'])}% | {fmt(l['gamma_above_0.999'])}% | "
                       + ' | '.join('–' if k is None else f'{k:.1e}' for k in kept)
                       + f" | {l['alpha']:.3f} | {', '.join(f'{x:.2f}' for x in l['sink_logits'])} | "
                       f"{l['phi_q_rms']:.3f} / {l['phi_q_max_abs']:.2f} | {l['phi_k_rms']:.3f} / "
                       f"{l['phi_k_max_abs']:.2f} | {l['W_gamma_norm']:.2f} |")
        out.append('')

    with open(join(run_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(out))
    scores = {m: {t: {k: v for k, v in r.items() if k not in ('questions', 'checkpoints', 'lizard')}
                  for t, r in runs[m].items()} for m in models}
    dump({'scores': scores, 'comparisons': comparisons, 'letters': letters,
          'checkpoints': checks, 'training': training, 'lizard': stats}, join(run_dir, 'summary.json'))
    print(f'-> Wrote {join(run_dir, "summary.md")} and summary.json')


def summarize_temperatures(root):
    """Compare the MMLU-subset runs in ROOT/T=<t>/ (docs/experiments/temperature.md)"""
    runs = {}
    for d in sorted(os.listdir(root)):
        if d.startswith('T=') and isdir(join(root, d)):
            for m in MODELS:
                if isfile(join(root, d, m, 'mmlu_subset', 'results.json')):
                    runs[(float(d[2:]), m)] = read_run(join(root, d, m, 'mmlu_subset'))
    rows, out = [], ['# Temperature on the MMLU subset', '',
                     f'Run directory: `{root}`. Each temperature divides the logits before the harness takes '
                     'log_softmax. "Changed" counts the questions whose predicted letter differs from the run at '
                     'temperature 1 of the same model.', '',
                     '| Temperature | Model | Tokens per letter | Accuracy | "A" | "B" | "C" | "D" | Changed | Mass '
                     '| Confidence | Entropy (bits) |', '|---' * 12 + '|']
    for (temperature, m), r in sorted(runs.items(), key=lambda kv: (list(MODELS).index(kv[0][1]), kv[0][0])):
        qs = r['questions']
        base = runs.get((1.0, m))
        changed = sum(qs[q]['prediction'] != base['questions'][q]['prediction']
                      for q in qs if q in base['questions']) if base else None
        predicted = Counter(q['prediction'] for q in qs.values())
        shares = [predicted[i] / len(qs) for i in range(4)]
        rows.append({'temperature': temperature, 'model': m, 'answer_letter_tokens': r['answer_letter_tokens'],
                     'acc': r['acc'], 'predicted': dict(zip(LETTERS, shares)), 'changed': changed,
                     'n': r['n'], **{k: r[f'mean_{k}'] for k in ('mass', 'confidence', 'entropy_bits')}})
        changed_text = '–' if changed is None else f"{changed} of {r['n']}"
        out.append(f"| {temperature:g} | {MODELS[m]} | {fmt_tokens(r['answer_letter_tokens'])} | {fmt(r['acc'])} | "
                   + ' | '.join(fmt(x) for x in shares)
                   + f" | {changed_text} | {r['mean_mass']:.3f} | {r['mean_confidence']:.3f} | "
                   f"{r['mean_entropy_bits']:.3f} |")
    out += ['', 'Prediction: with one token per answer letter, the predicted letters and the accuracy are the same '
            'at every temperature above 0. Only the mass, the confidence and the entropy change.', '']
    with open(join(root, 'summary.md'), 'w') as f:
        f.write('\n'.join(out))
    dump({'runs': rows}, join(root, 'summary.json'))
    print(f'-> Wrote {join(root, "summary.md")} and summary.json')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else None
    if command == 'eval' and len(sys.argv) > 2:
        run_eval(sys.argv[2], sys.argv[3:])
    elif command == 'summary' and len(sys.argv) == 3:
        summarize(sys.argv[2])
    elif command == 'temperatures' and len(sys.argv) == 3:
        summarize_temperatures(sys.argv[2])
    else:
        sys.exit(__doc__)
