"""
Sample attention weights of the teacher's softmax attention and of Lizard attention, from a saved checkpoint
(docs/experiments/xai-sample-attention-weight.md; like Figures 18-21 of the LoLCATs paper). No training.
-> compute: one checkpoint (stage 1, stage 2, or the initial Lizard weights) on held-out packed Alpaca samples
   python scripts/attention_weights.py compute OUT.json --model_config M --distill_config D [--stage 2] [options]
   - Samples: the stage 1 validation split (held out from training), packed into chunks of --seq_len tokens
   - Teacher: the unconverted model runs first. Forward pre-hooks keep the input of each attention layer.
     Its weights: softmax(q k^T / sqrt(d)) with RoPE, causal
   - Lizard: A = A_gla + alpha * A_window, so that A @ v is the Lizard output (checked in every layer):
     A_gla, the gated linear attention weights, normalized per query; A_window, the sliding window softmax,
     whose sinks take a part of the mass
   - --inputs teacher (default): each Lizard layer gets the input of the same teacher layer, so both maps
     use the same hidden states. --inputs own: the converted model runs on its own hidden states
   - Crops (the panels of the figure): for --layers x --heads, the last --queries queries against the first
     and the last --queries keys, for the teacher, Lizard and the gated branch of Lizard
   - Metrics for all layers and heads, over the full maps and the queries at or after the window: the mass
     on the first key, the mass outside the window, the total variation distance to the teacher (after
     normalizing the Lizard rows), the Lizard row sum, and the window branch's share and sink mass
   - --stage 1 --checkpoint init: no checkpoint, the Lizard weights at initialization (seed --seed)
-> plot: one PNG per head (rows: the teacher, then each RESULT.json; columns: the layers), a PNG of the
   metrics for all layers and heads, and a Markdown table of the metrics per layer
   python scripts/attention_weights.py plot OUT_PREFIX RESULT.json [RESULT.json ...] [--heads H ...] [--sample S]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import torch

from layer_mse import TEXT, TEXT_2, default_checkpoint, download, dump

METRICS = ('first_key', 'outside_window', 'tv_distance', 'row_sum', 'window_share', 'window_sink_mass')


def round_list(t, digits=4):
    """Nested lists of floats with `digits` significant digits, to keep the JSON small"""
    return [[float(f'{x:.{digits}g}') for x in row] for row in t.tolist()]


# --- compute ---

def capture_inputs(model, layers, run):
    """Run `run()` and return each attention layer's input, position data and output"""
    found = {}
    hooks = []
    for i, layer in enumerate(layers):
        def pre(module, args, kwargs, i=i):
            x = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            found[i] = {'x': x.detach(), 'position_ids': kwargs.get('position_ids'),
                        'position_embeddings': kwargs.get('position_embeddings')}

        def post(module, args, kwargs, output, i=i):
            found[i]['output'] = output[0].detach()
        hooks += [layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True),
                  layer.self_attn.register_forward_hook(post, with_kwargs=True)]
    with torch.no_grad():
        run()
    for h in hooks:
        h.remove()
    return found


def rel_error(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def teacher_qk(attn, captured):
    """
    q and k with RoPE of a Llama attention layer on its captured input (kept on the CPU), and the check:
    the softmax weights times v, through o_proj, against the layer's own output
    """
    from src.model.rotary import apply_rotary_pos_emb
    x = captured['x']
    b, n, _ = x.shape
    heads, kv_heads = attn.config.num_attention_heads, attn.config.num_key_value_heads
    d = attn.config.hidden_size // heads
    q = attn.q_proj(x).view(b, n, heads, d).transpose(1, 2)
    k = attn.k_proj(x).view(b, n, kv_heads, d).transpose(1, 2)
    v = attn.v_proj(x).view(b, n, kv_heads, d).transpose(1, 2)
    if captured['position_embeddings'] is not None:
        cos, sin = captured['position_embeddings']
    else:
        position_ids = captured['position_ids']
        if position_ids is None:
            position_ids = torch.arange(n, device=x.device)[None]
        cos, sin = attn.rotary_emb(v, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k, v = (t.repeat_interleave(heads // kv_heads, dim=1) for t in (k, v))
    y = (softmax_weights(q, k) @ v.float()).transpose(1, 2).reshape(b, n, heads * d)
    check = rel_error(attn.o_proj(y.to(x.dtype)), captured['output'])
    return q.cpu(), k.cpu(), check


def softmax_weights(q, k):
    """Causal softmax(q k^T / sqrt(d)) in fp32 (or fp64 for fp64 inputs)"""
    q, k = q.to(torch.promote_types(q.dtype, torch.float32)), k.to(torch.promote_types(k.dtype, torch.float32))
    n = q.shape[-2]
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    return scores.masked_fill(~causal, float('-inf')).softmax(-1)


def lizard_weights(attn, x):
    """Effective Lizard weights A (A @ v = the Lizard output before o_proj) and the gated branch weights"""
    from src.model.linear_attention.lizard_attention import gate_products, sink_softmax, upcast, window_mask
    q = attn.split(attn.q_proj(x), attn.heads)
    k = attn.split(attn.k_proj(x), attn.kv_heads)
    v = attn.split(attn.v_proj(x), attn.kv_heads)
    gamma = torch.sigmoid(*upcast(attn.W_gamma(x))).squeeze(-1)  # as in LolcatsLizardAttention.forward
    qf, kf, vf, alpha, meta = upcast(q, k, v, attn.alpha_blend, attn.meta_tokens)
    fq, fk = attn.feature_maps(qf, kf)
    w = (fq @ fk.transpose(-1, -2)) * gate_products(gamma)[:, None]
    a_gla = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)  # as in gla()
    n = q.shape[-2]
    scores = qf @ kf.transpose(-1, -2) / math.sqrt(q.shape[-1])
    a_window = sink_softmax(scores.masked_fill(~window_mask(n, attn.window, q.device), float('-inf')), meta)
    weights = a_gla + alpha * a_window
    check = rel_error(weights @ vf, attn.lizard(q, k, v, gamma).float())
    return weights, a_gla, a_window, alpha.item(), check


def metrics(teacher, lizard, a_window, alpha, window):
    """Per head, means over the samples and the queries at or after the window (these have keys outside it)"""
    n = teacher.shape[-1]
    i = torch.arange(n, device=teacher.device)
    outside = (i[None, :] <= i[:, None] - window)  # key t is outside the window of query i
    rows = slice(window, n)
    row_sum = lizard.sum(-1)
    normalized = lizard / row_sum[..., None]
    per_query = {
        'first_key': (teacher[..., 0], lizard[..., 0]),
        'outside_window': ((teacher * outside).sum(-1), (lizard * outside).sum(-1)),
    }
    out = {}
    for name, (t, l) in per_query.items():
        out[f'teacher_{name}'] = t[..., rows].mean(dim=(0, 2)).tolist()
        out[f'lizard_{name}'] = l[..., rows].mean(dim=(0, 2)).tolist()
    out['tv_distance'] = (0.5 * (teacher - normalized).abs().sum(-1))[..., rows].mean(dim=(0, 2)).tolist()
    out['row_sum'] = row_sum[..., rows].mean(dim=(0, 2)).tolist()
    window_mass = a_window.sum(-1)
    out['window_share'] = (alpha * window_mass / row_sum)[..., rows].mean(dim=(0, 2)).tolist()
    out['window_sink_mass'] = (1 - window_mass)[..., rows].mean(dim=(0, 2)).tolist()
    return out


def crop(weights, queries):
    """The last `queries` queries against the first and the last `queries` keys"""
    n = weights.shape[-1]
    cols = torch.cat([torch.arange(queries), torch.arange(n - queries, n)]).to(weights.device)
    return weights[..., n - queries:, :][..., cols]


def compute(args):
    sys.path.insert(0, '.')  # as distill_llama.py, run from the repo root
    sys.path.append('./src')
    from omegaconf import OmegaConf
    from compare_stages import check_checkpoint
    from dataloaders import load_data
    from model.convert_model import toggle_attention, traverse_layers
    from model.load_model import load_and_convert_attns
    from model.peft import create_peft_config
    from model.pretrained import get_pretrained_loader
    from utils.setup import seed_everything

    seed_everything(args.seed)
    model_config = OmegaConf.load(f'configs/model/{args.model_config}.yaml')
    distill_config = OmegaConf.load(f'configs/experiment/{args.distill_config}.yaml')
    if args.cache_dir is not None:
        model_config.model.cache_dir = args.cache_dir
    if args.torch_dtype is not None:
        model_config.model.torch_dtype = args.torch_dtype
        if args.torch_dtype == 'float32':  # FlashAttention-2 supports only fp16 and bf16
            model_config.model.attn_implementation = 'eager'
    for k in ['pretrained_model_name_or_path', 'cache_dir']:  # as distill_llama.py does
        distill_config.dataset.pretrained_model_config[k] = model_config.model[k]
    distill_config.dataset.dataset_config.chunk_size = args.seq_len

    stage1 = args.checkpoint or default_checkpoint(args)
    if stage1 != 'init':
        stage1 = download(stage1, args.hf_repo)
    stage2 = None
    if args.stage == 2:
        assert stage1 != 'init', 'Stage 2 needs a stage 1 checkpoint'
        stage2 = args.finetune_checkpoint or \
            f"{default_checkpoint(args)[:-len('_distill.pt')]}-se={args.seed}-re={args.replicate}_ft.pt"
        stage2 = download(stage2, args.hf_repo)

    # Samples: packed chunks of the held-out validation split
    loader = load_data(distill_config.dataset, distill_config.dataloader)[distill_config.trainer.val_split]
    samples = []
    for data in loader:
        if len(samples) == args.samples:
            break
        if data['input_ids'].shape[-1] == args.seq_len:
            samples.append(data['input_ids'])
    assert samples, f'No validation chunk with {args.seq_len} tokens'

    # Teacher: the unconverted model, with the inputs of each attention layer
    attention_type = model_config['attention']['attention_type']
    model = get_pretrained_loader(**model_config.model, huggingface_token=os.environ.get('HF_TOKEN')).load(
        model_type=attention_type)
    model.eval()
    device = next(model.parameters()).device
    n_layers = len(traverse_layers(model))
    layers = args.layers if args.layers else sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4,
                                                     n_layers - 1})
    n_heads = model.config.num_attention_heads
    assert all(0 <= l < n_layers for l in layers), f'--layers must be in 0..{n_layers - 1}'
    assert all(0 <= h < n_heads for h in args.heads), f'--heads must be in 0..{n_heads - 1}'
    assert args.queries <= args.seq_len // 2, '--queries must be at most half of --seq_len'
    teacher, teacher_checks = [], {}
    for ids in samples:
        found = capture_inputs(model, traverse_layers(model),
                               lambda: model(input_ids=ids.to(device), use_cache=False))
        sample = {}
        with torch.no_grad():
            for i, layer in enumerate(traverse_layers(model)):
                q, k, check = teacher_qk(layer.self_attn, found[i])
                sample[i] = {'q': q, 'k': k, 'x': found[i]['x'].cpu()}
                teacher_checks[i] = max(teacher_checks.get(i, 0.0), check)
        teacher.append(sample)
        del found

    # Lizard: convert this model, then load the checkpoints (as src/model/load_model_for_eval.py)
    model, _ = load_and_convert_attns(model, model_config, attention_type=attention_type,
                                      checkpoint_path=None if stage1 == 'init' else stage1,
                                      merge_loras=False, train_converted=False, train_attention=False)
    model = toggle_attention(model, train=False)
    checks = {'stage1': {'path': 'init', 'seed': args.seed} if stage1 == 'init'
              else check_checkpoint(model, stage1, 'stage1')}
    if stage2 is not None:
        finetune_config = OmegaConf.load(f'configs/experiment/{args.finetune_config}.yaml')
        model, _ = create_peft_config(model, finetune_config.finetune)
        keys = model.load_state_dict(torch.load(stage2, map_location='cpu')['model_state_dict'], strict=False)
        assert not keys.unexpected_keys, f'Unexpected keys in {stage2}: {keys.unexpected_keys[:5]}'
        checks['stage2'] = check_checkpoint(model, stage2, 'stage2')
        model = model.merge_and_unload()  # create_peft_config casts the model to bf16, as in the evaluation
    model.eval()
    for stage, c in checks.items():
        if 'expected_keys' in c:
            print(f"-> {stage} checkpoint: {c['expected_keys']} expected keys, {len(c['missing_keys'])} missing, "
                  f"{len(c['unexpected_keys'])} unexpected, {len(c['values_not_loaded'])} not loaded, "
                  f"step {c['step']}, losses {c['losses']}")
    inputs = [{i: layer['x'] for i, layer in sample.items()} for sample in teacher]  # --inputs teacher
    if args.inputs == 'own':
        inputs = []
        for ids in samples:
            found = capture_inputs(model, traverse_layers(model),
                                   lambda: model(input_ids=ids.to(device), use_cache=False))
            inputs.append({i: f['x'].cpu() for i, f in found.items()})

    # Weights, crops and metrics, one layer at a time
    lizard_layers = traverse_layers(model)
    window = lizard_layers[0].self_attn.window
    per_layer, crops = [], [{} for _ in samples]
    with torch.no_grad():
        for i in range(n_layers):
            attn = lizard_layers[i].self_attn
            found, lizard_check = [], 0.0
            for s in range(len(samples)):
                t = softmax_weights(teacher[s][i]['q'].to(device), teacher[s][i]['k'].to(device))
                l, a_gla, a_window, alpha, check = lizard_weights(attn, inputs[s][i].to(device=device,
                                                                                         dtype=attn.q_proj.weight.dtype))
                lizard_check = max(lizard_check, check)
                found.append(metrics(t, l, a_window, alpha, window))
                if i in layers:
                    ct, cl, cg = crop(t, args.queries), crop(l, args.queries), crop(a_gla, args.queries)
                    crops[s][str(i)] = {str(h): {'teacher': round_list(ct[0, h]), 'lizard': round_list(cl[0, h]),
                                                 'gla': round_list(cg[0, h])} for h in args.heads}
                del t, l, a_gla, a_window
            mean = {k: [sum(f[k][h] for f in found) / len(found) for h in range(len(found[0][k]))]
                    for k in found[0]}
            per_layer.append({'layer': i, 'alpha': alpha, 'teacher_check': teacher_checks[i],
                              'lizard_check': lizard_check, **mean})
            print(f"-> Layer {i}: TV distance {sum(mean['tv_distance']) / len(mean['tv_distance']):.3f}, "
                  f"outside the window: teacher {sum(mean['teacher_outside_window']) / len(mean['tv_distance']):.3f}, "
                  f"Lizard {sum(mean['lizard_outside_window']) / len(mean['tv_distance']):.3f}; "
                  f"checks: teacher {teacher_checks[i]:.1e}, Lizard {lizard_check:.1e}")

    result = {
        'label': args.label or ('Lizard (init.)' if stage1 == 'init' else f'Lizard, stage {args.stage}'),
        'stage': 0 if stage1 == 'init' else args.stage,
        'model_config': args.model_config,
        'distill_config': args.distill_config,
        'finetune_config': args.finetune_config if args.stage == 2 else None,
        'torch_dtype': str(model_config.model.torch_dtype),
        'inputs': args.inputs,
        'checkpoints': checks,
        'data': {'split': distill_config.trainer.val_split, 'seq_len': args.seq_len, 'samples': len(samples)},
        'window': window,
        'queries': args.queries,
        'n_layers': n_layers,
        'n_heads': lizard_layers[0].self_attn.heads,
        'crop_layers': layers,
        'crop_heads': args.heads,
        'layers': per_layer,  # metrics for every layer; per head lists
        'crops': crops,  # per sample: {layer: {head: {teacher, lizard, gla}}}
        'env': {'torch': torch.__version__, 'device': str(device),
                'lolcats_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                                 text=True).stdout.strip() or None},
    }
    dump(result, args.out)
    worst = max(max(l['teacher_check'], l['lizard_check']) for l in per_layer)
    print(f'-> Largest relative error of the checks (A @ v against the layer output): {worst:.1e}')
    print(f'-> Wrote {args.out}')


# --- plot ---

def mean(xs):
    return sum(xs) / len(xs)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))
    first = results[0]
    q, n = first['queries'], first['data']['seq_len']
    layers = [l for l in first['crop_layers'] if all(l in r['crop_layers'] for r in results)]
    heads = args.heads or [h for h in first['crop_heads'] if all(h in r['crop_heads'] for r in results)]
    names = ['Softmax (teacher)'] + [r['label'] for r in results]
    ticks = [t for t in range(0, q, 10)]

    for h in heads:
        rows = [[first['crops'][args.sample][str(l)][str(h)]['teacher'] for l in layers]]
        rows += [[r['crops'][args.sample][str(l)][str(h)]['lizard'] for l in layers] for r in results]
        fig, axes = plt.subplots(len(rows), len(layers), figsize=(2.9 * len(layers), 2.55 * len(rows)),
                                 squeeze=False)
        for c, l in enumerate(layers):
            vmax = max(max(max(row) for row in rows[r][c]) for r in range(len(rows)))  # shared within a column
            for r in range(len(rows)):
                ax = axes[r][c]
                ax.imshow(rows[r][c], cmap='Blues', vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
                ax.axvline(q - 0.5, color=TEXT, linestyle='--', linewidth=1)
                ax.set_title(f'{names[r]}\nLayer {l} Head {h}', color=TEXT, fontsize=9)
                ax.set_xticks(ticks + [q + t for t in ticks])
                ax.set_xticklabels([str(t) for t in ticks] + [str(n - q + t) for t in ticks], fontsize=6)
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(n - q + t) for t in ticks], fontsize=6)
                ax.tick_params(colors=TEXT_2, length=2)
                ax.set_xlabel('Keys', color=TEXT, fontsize=8)
                if c == 0:
                    ax.set_ylabel('Queries', color=TEXT, fontsize=8)
        fig.suptitle(f'Head {h}: the last {q} queries against the first and the last {q} keys of a {n}-token sample.'
                     f'\nColor scale: 0 to the largest value of each column.', color=TEXT_2, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2.55 * len(rows))))
        fig.savefig(f'{args.out}_head{h}.png', dpi=150, facecolor='white')
        plt.close(fig)
        print(f'-> Wrote {args.out}_head{h}.png')

    # Metrics of all layers and heads: where the teacher looks outside the window, and how far each result is
    panels = [('teacher_outside_window', first, 'Softmax (teacher)\nmass outside the window')] + \
             [('tv_distance', r, f"{r['label']}\nTV distance to the teacher") for r in results]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.2), squeeze=False)
    for ax, (key, r, title) in zip(axes[0], panels):
        image = ax.imshow([l[key] for l in r['layers']], cmap='Blues', vmin=0, vmax=1, aspect='auto',
                          interpolation='nearest')
        ax.set_title(title, color=TEXT, fontsize=9)
        ax.set_xlabel('Head', color=TEXT, fontsize=8)
        ax.set_ylabel('Layer', color=TEXT, fontsize=8)
        ax.set_yticks(range(r['n_layers']))
        ax.set_xticks(range(0, r['n_heads'], max(1, r['n_heads'] // 8)))
        ax.tick_params(colors=TEXT_2, labelsize=7)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label='0 to 1')
    fig.savefig(f'{args.out}_metrics.png', dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)

    # The same metrics as tables: means over the heads of each layer, and the heads farthest from the teacher
    out = [f'Means over the {first["n_heads"]} heads of each layer, over the queries at or after the window '
           f'({first["window"]} tokens). Samples: {first["data"]["samples"]} of {n} tokens, inputs: '
           f'{first["inputs"]}.', '',
           '| Layer | Teacher: first key | Teacher: outside the window | '
           + ' | '.join(f"{r['label']}: outside the window | {r['label']}: TV distance | "
                        f"{r['label']}: window share" for r in results) + ' |',
           '|---' * (3 + 3 * len(results)) + '|']
    for i in range(first['n_layers']):
        cells = [f"{mean(first['layers'][i]['teacher_first_key']):.3f}",
                 f"{mean(first['layers'][i]['teacher_outside_window']):.3f}"]
        for r in results:
            l = r['layers'][i]
            cells += [f"{mean(l['lizard_outside_window']):.3f}", f"{mean(l['tv_distance']):.3f}",
                      f"{mean(l['window_share']):.3f}"]
        out.append(f'| {i} | ' + ' | '.join(cells) + ' |')
    for r in results:
        pairs = sorted(((l['tv_distance'][h], l['layer'], h) for l in r['layers'] for h in range(r['n_heads'])),
                       reverse=True)[:args.top]
        out += ['', f"{r['label']}: the {args.top} heads with the largest TV distance", '',
                '| Layer | Head | TV distance | Teacher: outside the window | Lizard: outside the window | '
                'Teacher: first key | Lizard: first key |', '|---|---|---|---|---|---|---|']
        for tv, i, h in pairs:
            l = r['layers'][i]
            out.append(f"| {i} | {h} | {tv:.3f} | {l['teacher_outside_window'][h]:.3f} | "
                       f"{l['lizard_outside_window'][h]:.3f} | {l['teacher_first_key'][h]:.3f} | "
                       f"{l['lizard_first_key'][h]:.3f} |")
        worst = max(max(l['teacher_check'], l['lizard_check']) for l in r['layers'])
        out += ['', f"Largest relative error of the checks: {worst:.1e}"]
    with open(f'{args.out}_metrics.md', 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'-> Wrote {args.out}_metrics.png and {args.out}_metrics.md')


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('compute')
    c.add_argument('out', help='Output JSON file')
    c.add_argument('--model_config', required=True)
    c.add_argument('--distill_config', required=True)
    c.add_argument('--finetune_config', default='finetune_lora_qkvo_alpaca_clean_1b',
                   help='The checkpoint names, and the LoRA setup of stage 2')
    c.add_argument('--stage', type=int, choices=(1, 2), default=1)
    c.add_argument('--checkpoint', default=None, help='Stage 1 checkpoint, or `init`. Default: the name of '
                                                      'distill_llama.py for the configs')
    c.add_argument('--finetune_checkpoint', default=None, help='Stage 2 checkpoint (default: as compare_stages.sh)')
    c.add_argument('--hf_repo', default=os.environ.get('HF_REPO', 'nanoman1/lolcats-lizard-llama-3.2-1b'))
    c.add_argument('--label', default=None, help='Name of this result in the plots')
    c.add_argument('--layers', type=int, nargs='+', default=None,
                   help='Layers of the crops (default: 0, 1/4, 1/2, 3/4 and the last layer)')
    c.add_argument('--heads', type=int, nargs='+', default=[0, 1, 2, 3], help='Heads of the crops')
    c.add_argument('--queries', type=int, default=32, help='Size of the crops: the last N queries, the first '
                                                           'and the last N keys')
    c.add_argument('--seq_len', type=int, default=1024, help='Tokens per packed sample')
    c.add_argument('--samples', type=int, default=1)
    c.add_argument('--inputs', choices=('teacher', 'own'), default='teacher')
    c.add_argument('--torch_dtype', default=None, help='Replaces the model config dtype, e.g. float32')
    c.add_argument('--cache_dir', default=None, help="Replaces the model config's cache_dir")
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--replicate', type=int, default=0)
    p = commands.add_parser('plot')
    p.add_argument('out', help='Output prefix: writes OUT_head<h>.png, OUT_metrics.png and OUT_metrics.md')
    p.add_argument('results', nargs='+', help='JSON files of compute; the teacher row comes from the first one')
    p.add_argument('--heads', type=int, nargs='+', default=None, help='Default: all heads with crops')
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--top', type=int, default=10, help='Rows of the table of the heads farthest from the teacher')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    compute(args) if args.command == 'compute' else plot(args)
