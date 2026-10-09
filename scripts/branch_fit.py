"""
Branch decomposition of the stage 1 attention (X2 of docs/20-layer-mse-for-piqa-arc.md,
docs/experiments/xai-branch-fit.md). No training: forward passes only.
-> compute: one stage 1 checkpoint (Lizard v1 or v2) on the stage 1 validation data, written to OUT (JSON)
   python scripts/branch_fit.py compute OUT --from_json docs/experiments/xai-layer-wise-mse/config1.json
   python scripts/branch_fit.py compute OUT --model_config M --distill_config D [options]
   - The model, data and checkpoint of scripts/layer_mse.py (the same options). Each layer gets the teacher input
   - For each layer and head, from the same q, k and v (outputs before o_proj):
       y        the teacher: softmax attention with RoPE over all previous keys (the stage 1 target)
       G        the gated branch of the checkpoint (Hedgehog maps, gate and its normalization)
       W        the window branch of the checkpoint (v1: no RoPE, with the sinks), without the factor alpha
       WR       an oracle: softmax attention with RoPE over the same window of keys, without sinks
       trained  the output of the checkpoint (v1: G + alpha W)
   - For each set of candidates (G, W, WR, G+W, G+WR, W+WR, G+W+WR) and each head: the least-squares weights c that
     minimize ||y - sum_s c_s y_s||^2 over the positions of a bucket, and the remaining error / ||y||^2. With one
     free weight for each branch and head, no scale of these branches (e.g., alpha) gives a lower error
   - Buckets of query positions: --edges (default 0,128,512,2048), and all positions. A query before position
     `window` sees all its keys inside the window, so WR is the teacher there: its error is only rounding
   - Per layer: the error over all heads (each head with its own weights), the error of each head, and the weights
   - Check: the branches reproduce the output of the checkpoint (G + alpha W, or the v2 branch weights times v)
-> plot: heatmaps of layers x candidate sets for one bucket, one panel per result, written to OUT.png and OUT.md
   python scripts/branch_fit.py plot OUT RESULT.json [RESULT.json ...] [--bucket 0-127]
   - One color scale for all panels; a log scale (at most 3 decades) if the values span more than a factor of 20
"""
import argparse
import json
import math
import os
import subprocess

import torch

from layer_mse import LABEL_HELP, TEXT, TEXT_2, dump, load
from position_mse import RAMP

# Index 0: the teacher output y; 1-3: the candidates; 4: the output of the checkpoint
CANDIDATES = ['G', 'W', 'WR']
SETS = {'G': [1], 'W': [2], 'WR': [3], 'G+W': [1, 2], 'G+WR': [1, 3], 'W+WR': [2, 3], 'G+W+WR': [1, 2, 3]}
COLUMNS = list(SETS) + ['trained']


def short(x, digits=6):
    return float(f'{x:.{digits}g}')


def rel_error(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# --- compute ---

def branches(attn, x, kwargs):
    """G, W (without alpha), WR, the checkpoint output from the branches, and alpha of each head"""
    from src.model.linear_attention.lizard_attention import awa, gla, upcast, window_mask
    from src.model.rotary import apply_rotary_pos_emb
    assert hasattr(attn, 'W_gamma'), f'{type(attn).__name__}: branch_fit.py supports Lizard layers (v1 and v2) only'
    q = attn.split(attn.q_proj(x), attn.heads)
    k = attn.split(attn.k_proj(x), attn.kv_heads)
    v = attn.split(attn.v_proj(x), attn.kv_heads)
    rope = kwargs.get('position_embeddings')
    if rope is None:  # as the layer itself does
        position_ids = kwargs.get('position_ids')
        if position_ids is None:
            position_ids = torch.arange(v.shape[-2], device=v.device)[None]
        rope = attn.rotary_emb(v, position_ids)
    if hasattr(attn, 'branch_weights'):  # v2: the output is (a_gla + a_win) @ v, alpha inside a_win
        gamma = torch.sigmoid(*upcast(attn.W_gamma(x))).transpose(1, 2)
        a_gla, a_win = attn.branch_weights(q, k, gamma, rope)
        vf = v.to(a_gla.dtype)
        alpha = attn.lizard_params(a_gla.dtype)[0].reshape(-1).expand(attn.heads)
        gated, window_alpha = a_gla @ vf, a_win @ vf
        safe = torch.where(alpha.abs() > 0, alpha, torch.ones_like(alpha))[None, :, None, None]
        window = window_alpha / safe
    else:  # v1: the output is G + alpha W
        gamma = torch.sigmoid(*upcast(attn.W_gamma(x))).squeeze(-1)
        qf, kf, vf, alpha, meta = upcast(q, k, v, attn.alpha_blend, attn.meta_tokens)
        fq, fk = attn.feature_maps(qf, kf)
        gated, window = gla(fq, fk, vf, gamma), awa(qf, kf, vf, meta, attn.window)
        window_alpha = alpha * window
        alpha = alpha.reshape(-1).expand(attn.heads)
    q_rope, k_rope = upcast(*apply_rotary_pos_emb(q, k, *rope))  # the teacher's q and k
    n = q.shape[-2]
    scores = q_rope @ k_rope.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~window_mask(n, attn.window, q.device), float('-inf'))
    window_rope = scores.softmax(-1) @ vf
    return gated, window, window_rope, gated + window_alpha, alpha.tolist()


def solve(gram):
    """gram (h, 5, 5): for each candidate set, the least-squares weights of each head and the remaining errors"""
    yy = gram[:, 0, 0]
    fits = {}
    for name, idx in SETS.items():
        a = gram[:, idx][:, :, idx]
        r = gram[:, idx, 0]
        c = (torch.linalg.pinv(a, hermitian=True) @ r.unsqueeze(-1)).squeeze(-1)  # stable if a is singular
        fits[name] = {'error': (yy - (c * r).sum(-1)).clamp_min(0), 'weights': c}
    fits['trained'] = {'error': (yy - 2 * gram[:, 0, 4] + gram[:, 4, 4]).clamp_min(0)}
    return fits, yy


def compute(args):
    if args.from_json is not None:
        with open(args.from_json) as f:
            ref = json.load(f)
        args.model_config = args.model_config or ref['model_config']
        args.distill_config = args.distill_config or ref['distill_config']
        args.checkpoint = args.checkpoint or ref['checkpoint'].get('path')
        args.label = args.label or ref['label']
    assert args.model_config and args.distill_config, 'Pass --from_json, or --model_config and --distill_config'
    model, model_config, distill_config, check, loader = load(args)

    layers = model.model.layers
    state = {'bounds': None, 'gram': [None] * len(layers), 'check': [0.0] * len(layers), 'alpha': [None] * len(layers)}

    def make_hook(i):
        def hook(module, hook_args, kwargs, output):
            x = kwargs['hidden_states'] if 'hidden_states' in kwargs else hook_args[0]
            y_pred, y_true = output[1][1]
            gated, window, window_rope, combined, alpha = branches(module, x, kwargs)
            state['check'][i] = max(state['check'][i], rel_error(combined, y_pred))
            state['alpha'][i] = alpha
            stack = torch.stack([t.double() for t in (y_true, gated, window, window_rope, y_pred)])  # (5, b, h, l, d)
            grams = torch.stack([torch.einsum('sbhld,tbhld->hst', stack[:, :, :, lo:hi], stack[:, :, :, lo:hi])
                                 for lo, hi in state['bounds']]).cpu()  # (buckets, h, 5, 5)
            state['gram'][i] = grams if state['gram'][i] is None else state['gram'][i] + grams
        return hook

    hooks = [layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True) for i, layer in enumerate(layers)]
    batches, tokens, length = 0, 0, None
    with torch.no_grad():
        for ix, data in enumerate(loader):
            if args.max_batches is not None and ix == args.max_batches:
                break
            inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}
            if length is None:
                length = inputs['input_ids'].shape[1]
                edges = sorted({min(e, length) for e in args.edges} | {0, length})
                state['bounds'] = list(zip(edges[:-1], edges[1:]))
                print(f"-> Sequence length {length}, buckets {[f'{lo}-{hi - 1}' for lo, hi in state['bounds']]}")
            assert inputs['input_ids'].shape[1] == length, 'All validation sequences must have the same length'
            model(**inputs, output_attentions=True, use_cache=False)
            batches += 1
            tokens += inputs['input_ids'].numel()
            print(f'-> Batch {ix + 1} done')
    for h in hooks:
        h.remove()
    assert batches > 0, 'No validation batches'

    names = [f'{lo}–{hi - 1}' for lo, hi in state['bounds']] + ['all']
    result_layers = []
    for i, grams in enumerate(state['gram']):
        grams = torch.cat([grams, grams.sum(0, keepdim=True)])  # the last bucket: all positions
        fits = {}
        for name, gram in zip(names, grams):
            solved, yy = solve(gram)
            fits[name] = {s: {'relative_error': (f['error'].sum() / yy.sum()).item(),
                              'per_head': [short(e) for e in (f['error'] / yy).tolist()],
                              **({'weights': [[short(c) for c in row] for row in f['weights'].tolist()]}
                                 if 'weights' in f else {})}
                          for s, f in solved.items()}
        result_layers.append({'layer': i, 'alpha': [short(a) for a in state['alpha'][i]],
                              'check_relative_error': state['check'][i], 'fits': fits})
    summary = {name: {s: sum(l['fits'][name][s]['relative_error'] for l in result_layers) / len(result_layers)
                      for s in COLUMNS} for name in names}
    result = {
        'label': args.label or args.distill_config,
        'model_config': args.model_config,
        'distill_config': args.distill_config,
        'torch_dtype': str(model_config.model.torch_dtype),
        'checkpoint': check or {'path': 'init', 'seed': args.seed},
        'data': {'split': distill_config.trainer.val_split, 'batches': batches, 'tokens': tokens,
                 'sequence_length': length, 'all_batches': args.max_batches is None},
        'candidates': {'G': 'gated branch', 'W': 'window branch of the checkpoint, without alpha',
                       'WR': 'oracle: softmax with RoPE over the window, without sinks'},
        'buckets': names,
        'summary': summary,  # the mean over the layers of the relative error of each layer
        'max_check_relative_error': max(state['check']),
        'layers': result_layers,
        'env': {'torch': torch.__version__, 'device': str(model.device),
                'lolcats_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                                 text=True).stdout.strip() or None},
    }
    dump(result, args.out)
    print('\nRemaining error / ||y||^2 after the least-squares fit, mean over the layers')
    print(f'{"Bucket":>10} ' + ' '.join(f'{c:>9}' for c in COLUMNS))
    for name in names:
        print(f'{name:>10} ' + ' '.join(f'{summary[name][c]:>9.4f}' for c in COLUMNS))
    print(f"-> Check: the branches reproduce the checkpoint output, largest relative error "
          f"{result['max_check_relative_error']:.2e}")
    print(f'-> Wrote {args.out}')


# --- plot ---

def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    bucket = args.bucket.replace('-', '–')
    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))
    for r in results:
        assert bucket in r['buckets'], f"{r['label']}: no bucket {bucket} (buckets: {r['buckets']})"
    grids = [[[l['fits'][bucket][c]['relative_error'] for c in COLUMNS] for l in r['layers']] for r in results]
    values = [v for g in grids for row in g for v in row]
    low, high = min([v for v in values if v > 0] or [1.0]), max(values)
    if high / low > 20:  # log scale over at most 3 decades: smaller values get the lightest color
        floor = max(low, high * 1e-3)
        norm = LogNorm(floor, high)
    else:
        floor, norm = 0.0, Normalize(0, high)
    cmap = LinearSegmentedColormap.from_list('ramp', RAMP)
    title = (f'Remaining error / ||y||² after the least-squares fit of each candidate set, query positions {bucket}'
             if bucket != 'all' else 'Remaining error / ||y||² after the least-squares fit of each candidate set')

    ncols = min(3, len(results))
    nrows = math.ceil(len(results) / ncols)
    n_layers = max(len(r['layers']) for r in results)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                             figsize=(ncols * (1.0 + 0.62 * len(COLUMNS)) + 1.4, nrows * (1.2 + 0.27 * n_layers)))
    for k, (r, g) in enumerate(zip(results, grids)):
        ax = axes[k // ncols][k % ncols]
        image = ax.imshow([[max(v, floor) for v in row] for row in g], cmap=cmap, norm=norm, aspect='auto')
        for i, row in enumerate(g):
            for j, v in enumerate(row):  # dark ink on light cells, white ink on dark cells
                ax.text(j, i, f'{v:.2g}', ha='center', va='center', fontsize=6.5,
                        color='white' if norm(max(v, floor)) > 0.5 else TEXT)
        ax.set_xticks(range(len(COLUMNS)))
        ax.set_xticklabels(COLUMNS, fontsize=7.5, color=TEXT_2, rotation=45, ha='right')
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([str(l['layer']) for l in r['layers']], fontsize=8, color=TEXT_2)
        ax.set_xticks([x - 0.5 for x in range(1, len(COLUMNS))], minor=True)  # 2px gaps between cells
        ax.set_yticks([y - 0.5 for y in range(1, len(g))], minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='both', length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(r['label'], color=TEXT, fontsize=9)
        if k % ncols == 0:
            ax.set_ylabel('Layer', color=TEXT, fontsize=9)
    for k in range(len(results), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle(title, color=TEXT, fontsize=10)
    bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, extend='min' if low < floor else 'neither')
    bar.set_label('Relative error', color=TEXT, fontsize=8)
    bar.ax.tick_params(colors=TEXT_2, labelsize=8)
    bar.outline.set_visible(False)
    fig.savefig(f'{args.out}.png', dpi=200, facecolor='white')

    # The same values as tables, for the experiment documents
    lines = [title, '', 'G: gated branch. W: window branch of the checkpoint. WR: window with RoPE (oracle). '
             'trained: the output of the checkpoint.', '']
    for r, g in zip(results, grids):
        lines += [f"**{r['label']}**", '', '| Layer | ' + ' | '.join(COLUMNS) + ' |', '|---' * (len(COLUMNS) + 1) + '|']
        lines += [f"| {l['layer']} | " + ' | '.join(f'{v:.4g}' for v in row) + ' |' for l, row in zip(r['layers'], g)]
        lines += ['| Mean | ' + ' | '.join(f"{r['summary'][bucket][c]:.4g}" for c in COLUMNS) + ' |', '']
    with open(f'{args.out}.md', 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'-> Wrote {args.out}.png and {args.out}.md')


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('compute')
    c.add_argument('out', help='Output JSON file')
    c.add_argument('--from_json', default=None, help='A layer_mse.py result: its configs, checkpoint and label')
    c.add_argument('--model_config', default=None)
    c.add_argument('--distill_config', default=None)
    c.add_argument('--finetune_config', default='finetune_lora_qkvo_alpaca_clean_1b',
                   help='Only for the default checkpoint name')
    c.add_argument('--checkpoint', default=None, help='Stage 1 checkpoint, or `init` (see scripts/layer_mse.py)')
    c.add_argument('--hf_repo', default=os.environ.get('HF_REPO', 'nanoman1/lolcats-lizard-llama-3.2-1b'))
    c.add_argument('--label', default=None, help=LABEL_HELP)
    c.add_argument('--edges', default=[0, 128, 512, 2048], type=lambda s: [int(x) for x in s.split(',')],
                   help='Bucket edges in query positions, comma-separated (default: 0,128,512,2048)')
    c.add_argument('--torch_dtype', default=None, help='Replaces the model config dtype, e.g. float32')
    c.add_argument('--max_batches', type=int, default=None, help='Default: all validation batches')
    c.add_argument('--cache_dir', default=None, help="Replaces the model config's cache_dir")
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--replicate', type=int, default=0)
    p = commands.add_parser('plot')
    p.add_argument('out', help='Output path without extension: writes OUT.png and OUT.md')
    p.add_argument('results', nargs='+', help='JSON files of compute')
    p.add_argument('--bucket', default='all', help='Bucket of query positions, e.g. 0-127 (default: all)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    compute(args) if args.command == 'compute' else plot(args)
