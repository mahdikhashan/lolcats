"""
MSE between the teacher's softmax attention and Lizard attention for each bucket of query positions
(X1 of docs/20-layer-mse-for-piqa-arc.md, docs/experiments/xai-position-mse.md). No training: forward passes only.
-> compute: one stage 1 checkpoint on the stage 1 validation data, written to OUT (JSON)
   python scripts/position_mse.py compute OUT --from_json docs/experiments/xai-layer-wise-mse/config1.json
   python scripts/position_mse.py compute OUT --model_config M --distill_config D [options]
   - The same model, data and checkpoint as scripts/layer_mse.py compute (the same options). --from_json takes the
     model config, the distill config, the checkpoint path and the label of a layer_mse.py result
   - Positions count from the start of each validation sequence (2048 tokens). --edges 0,128,512,2048 (default)
     gives the buckets 0-127, 128-511 and 512-2047. In bucket 0-127, every key is inside the window of 128 tokens,
     as in almost all PIQA and ARC-Easy prompts. Edges above the sequence length are cut to it
   - Per layer and bucket: the MSE, the mean square of the teacher output, their ratio (relative MSE) and the
     MSE of each head. Means over the validation batches, as in layer_mse.py
   - Per bucket: mse_factor x the mean MSE over the layers (the scale of the stage 1 loss) and the mean relative MSE
   - Check: for each layer, the token-weighted mean of the bucket MSEs equals the MSE over all positions
-> rank: rank correlation (Spearman) between the score of each bucket and the accuracy, over several checkpoints
   python scripts/position_mse.py rank OUT RESULT.json [RESULT.json ...] --accuracy ACCURACY.csv [--exclude LABEL]
   - ACCURACY.csv has the columns label, piqa and arc_easy (an empty cell: not measured). The label of each
     result selects its row. --exclude drops results by label, e.g. the LoLCATs control
   - Scores: the loss of each bucket, and the stage 1 loss (all positions) for comparison. A negative rank
     correlation means: a lower MSE comes with a higher accuracy
   - p: the two-sided permutation p-value of the rank correlation (exact up to 9 checkpoints)
   - Writes OUT.md (the tables) and OUT.png (the accuracy against the loss of each bucket)
-> heatmap: layers x buckets for each result, one color scale for all panels, written to OUT.png and OUT.md
   python scripts/position_mse.py heatmap OUT RESULT.json [RESULT.json ...] [--absolute]
   - Default: the relative MSE (comparable between layers). --absolute: mse_factor x MSE
   - A log color scale if the values span more than a factor of 20 (e.g., with the LoLCATs control)
   - For a finer grid, compute with more edges, e.g. --edges 0,64,128,256,512,1024,2048
"""
import argparse
import csv
import itertools
import json
import math
import os
import random
import subprocess

import torch

from layer_mse import COLORS, GRID, LABEL_HELP, TEXT, TEXT_2, dump, load

TASKS = {'piqa': 'PIQA', 'arc_easy': 'ARC-Easy'}
# Sequential ramp for magnitudes: one hue (blue), from light (near 0) to dark
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def bucket_name(b):
    return f"{b['start']}–{b['end'] - 1}"


# --- compute ---

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
    criterion = torch.nn.MSELoss(reduction='mean')  # as in src/trainer/distill_attention_xent_mse.py

    sums, batches, tokens, length, bounds = None, 0, 0, None, None
    with torch.no_grad():
        for ix, data in enumerate(loader):
            if args.max_batches is not None and ix == args.max_batches:
                break
            inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}
            if length is None:
                length = inputs['input_ids'].shape[1]
                edges = sorted({min(e, length) for e in args.edges} | {0, length})
                bounds = list(zip(edges[:-1], edges[1:]))
                print(f'-> Sequence length {length}, buckets {[f"{lo}-{hi - 1}" for lo, hi in bounds]}')
            assert inputs['input_ids'].shape[1] == length, 'All validation sequences must have the same length'
            attentions = model(**inputs, output_attentions=True, use_cache=False).get('attentions')
            values = []
            for layer, attns in enumerate(attentions):  # (..., (y_pred, y_true)), shape (b, h, l, d)
                assert attns is not None, f'Layer {layer} has no distillation outputs (softmax_attentions)'
                y_pred, y_true = attns[1]
                error = (y_pred.float() - y_true.float()).pow(2)
                target = y_true.float().pow(2)
                row = [criterion(y_pred, y_true).item(), target.mean().item(), error.mean().item()]
                for lo, hi in bounds:
                    row += [error[:, :, lo:hi].mean().item(), target[:, :, lo:hi].mean().item()]
                    row += error[:, :, lo:hi].mean(dim=(0, 2, 3)).tolist()
                values.append(row)
            sums = values if sums is None else [[s + v for s, v in zip(a, b)] for a, b in zip(sums, values)]
            batches += 1
            tokens += inputs['input_ids'].numel()
            print(f'-> Batch {ix + 1}: {distill_config.trainer.mse_factor} x mean layer MSE = '
                  f'{distill_config.trainer.mse_factor * sum(v[0] for v in values) / len(values):.4f}')
    assert batches > 0, 'No validation batches'

    heads = (len(sums[0]) - 3) // len(bounds) - 2
    layers, worst = [], 0.0
    for i, s in enumerate(sums):
        s = [x / batches for x in s]
        mse, target_ms, mse_fp32 = s[:3]
        buckets = []
        for j, (lo, hi) in enumerate(bounds):
            b_mse, b_ms, *b_heads = s[3 + j * (heads + 2):3 + (j + 1) * (heads + 2)]
            buckets.append({'start': lo, 'end': hi, 'mse': b_mse, 'target_mean_square': b_ms,
                            'relative_mse': b_mse / b_ms, 'mse_per_head': b_heads})
        # Check: the bucket MSEs, weighted by the number of positions, give the MSE over all positions
        weighted = sum((b['end'] - b['start']) * b['mse'] for b in buckets) / length
        worst = max(worst, abs(weighted - mse_fp32) / mse_fp32)
        layers.append({'layer': i, 'mse': mse, 'target_mean_square': target_ms, 'relative_mse': mse / target_ms,
                       'buckets': buckets})
    mse_factor = float(distill_config.trainer.mse_factor)
    loss = mse_factor * sum(l['mse'] for l in layers) / len(layers)
    stored = (check or {}).get('losses', {}).get(distill_config.trainer.metric_for_best_model)
    summary = [{'start': lo, 'end': hi,
                'loss': mse_factor * sum(l['buckets'][j]['mse'] for l in layers) / len(layers),
                'relative_mse': sum(l['buckets'][j]['relative_mse'] for l in layers) / len(layers)}
               for j, (lo, hi) in enumerate(bounds)]
    result = {
        'label': args.label or args.distill_config,
        'model_config': args.model_config,
        'distill_config': args.distill_config,
        'torch_dtype': str(model_config.model.torch_dtype),
        'checkpoint': check or {'path': 'init', 'seed': args.seed},
        'data': {'split': distill_config.trainer.val_split, 'batches': batches, 'tokens': tokens,
                 'sequence_length': length, 'all_batches': args.max_batches is None},
        'mse_factor': mse_factor,
        'loss': loss,  # same scale and data as distill/eval/loss
        'stored_loss': stored,
        'buckets': summary,
        'check_max_relative_difference': worst,
        'layers': layers,
        'env': {'torch': torch.__version__, 'device': str(model.device),
                'lolcats_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                                 text=True).stdout.strip() or None},
    }
    dump(result, args.out)

    names = [bucket_name(b) for b in summary]
    print(f'\n{mse_factor:g} x MSE and relative MSE of each layer, for each bucket of query positions')
    print(f'{"Layer":>5} ' + ' '.join(f'{n:>22}' for n in names))
    for l in layers:
        print(f"{l['layer']:>5} " + ' '.join(f"{mse_factor * b['mse']:>12.4g} {b['relative_mse']:>9.4f}"
                                            for b in l['buckets']))
    print(f'{"Mean":>5} ' + ' '.join(f"{b['loss']:>12.4g} {b['relative_mse']:>9.4f}" for b in summary))
    print(f'-> {mse_factor:g} x mean layer MSE = {loss:.4f} over {batches} batches ({tokens} tokens); '
          f'stored loss in the checkpoint: {stored}')
    if stored is not None and args.max_batches is None:
        print(f'-> Difference to the stored loss: {100 * (loss - stored) / stored:+.3f}%')
    print(f'-> Check: weighted mean of the buckets against all positions, largest relative difference {worst:.2e}')
    assert worst < 1e-4, 'The bucket MSEs do not add up to the MSE over all positions'
    print(f'-> Wrote {args.out}')


# --- rank ---

def ranks(xs):
    """Ranks from 1, with the mean rank for ties"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    var = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return cov / var if var > 0 else float('nan')


def spearman(x, y, samples=20000, seed=0):
    """Rank correlation and its two-sided permutation p-value (exact up to 9 values)"""
    rx, ry = ranks(x), ranks(y)
    rho = pearson(rx, ry)
    if math.isnan(rho):
        return rho, float('nan')
    if len(x) <= 9:
        perms = list(itertools.permutations(ry))
    else:
        rng = random.Random(seed)
        perms = [rng.sample(ry, len(ry)) for _ in range(samples)]
    hits = sum(abs(pearson(rx, list(p))) >= abs(rho) - 1e-12 for p in perms)
    return rho, hits / len(perms)


def read_accuracy(path):
    table = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            table[row['label']] = {t: float(row[t]) if row.get(t, '').strip() else None for t in TASKS}
    return table


def rank(args):
    results = []
    for path in args.results:
        with open(path) as f:
            r = json.load(f)
        if r['label'] in args.exclude:
            continue
        results.append(r)
    accuracy = read_accuracy(args.accuracy)
    missing = [r['label'] for r in results if r['label'] not in accuracy]
    assert not missing, f'No row in {args.accuracy} for: {missing}'
    names = [bucket_name(b) for b in results[0]['buckets']]
    assert all([bucket_name(b) for b in r['buckets']] == names for r in results), 'All results need the same buckets'
    scores = {f'Positions {n}': [r['buckets'][j]['loss'] for r in results] for j, n in enumerate(names)}
    scores['All positions (stage 1 loss)'] = [r['loss'] for r in results]

    lines = [f'Results: {len(results)}' + (f" (excluded: {', '.join(args.exclude)})" if args.exclude else ''), '',
             '| # | Checkpoint | ' + ' | '.join(scores) + ' | ' + ' | '.join(TASKS.values()) + ' |',
             '|---' * (len(scores) + len(TASKS) + 2) + '|']
    for i, r in enumerate(results):
        acc = accuracy[r['label']]
        lines.append(f"| {i + 1} | {r['label']} | " + ' | '.join(f'{v[i]:.4g}' for v in scores.values()) + ' | '
                     + ' | '.join('–' if acc[t] is None else f'{acc[t]:.1f}' for t in TASKS) + ' |')
    lines += ['', 'Rank correlation (Spearman) between the loss and the accuracy. Below 0: a lower loss comes with '
              'a higher accuracy. p: two-sided permutation p-value.', '',
              '| Score | ' + ' | '.join(f'{name} ρ | {name} p | {name} n' for name in TASKS.values()) + ' |',
              '|---' * (3 * len(TASKS) + 1) + '|']
    for name, values in scores.items():
        cells = []
        for t in TASKS:
            pairs = [(v, accuracy[r['label']][t]) for v, r in zip(values, results) if accuracy[r['label']][t] is not None]
            if len(pairs) < 3:
                cells += ['–', '–', str(len(pairs))]
                continue
            rho, p = spearman([a for a, _ in pairs], [b for _, b in pairs])
            cells += [f'{rho:+.2f}', f'{p:.2g}', str(len(pairs))]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(f'{args.out}.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fig, axes = plt.subplots(len(TASKS), len(scores), figsize=(3.4 * len(scores), 3.0 * len(TASKS)), squeeze=False)
    for row, (t, task) in enumerate(TASKS.items()):
        for col, (name, values) in enumerate(scores.items()):
            ax = axes[row][col]
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(TEXT_2)
            ax.tick_params(colors=TEXT_2, labelsize=8)
            points = [(v, accuracy[r['label']][t], i) for i, (v, r) in enumerate(zip(values, results))
                      if accuracy[r['label']][t] is not None]
            for v, a, i in points:
                ax.scatter([v], [a], color=COLORS[i % len(COLORS)], s=36, edgecolors='white', linewidths=1, zorder=3)
                ax.annotate(str(i + 1), (v, a), textcoords='offset points', xytext=(4, 3), fontsize=8, color=TEXT)
            xs = [v for v, _, _ in points]
            if xs and min(xs) > 0 and max(xs) / min(xs) > 20:  # log axis only for a wide range
                ax.set_xscale('log')
                subs = (1.0, 2.0, 5.0) if math.log10(max(xs) / min(xs)) <= 2.5 else (1.0,)
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
            if row == 0:
                ax.set_title(name, color=TEXT, fontsize=9)
            if row == len(TASKS) - 1:
                ax.set_xlabel(f"{results[0]['mse_factor']:g} × mean layer MSE", color=TEXT, fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{task} accuracy', color=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{args.out}.png', dpi=200, facecolor='white')
    print(f'-> Wrote {args.out}.md and {args.out}.png (the numbers are the rows of the first table)')


# --- heatmap ---

def heatmap(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))
    key = 'mse' if args.absolute else 'relative_mse'
    grids = [[[(r['mse_factor'] if args.absolute else 1.0) * b[key] for b in l['buckets']] for l in r['layers']]
             for r in results]
    values = [v for g in grids for row in g for v in row]
    low, high = min([v for v in values if v > 0] or [1.0]), max(values)
    if high / low > 20:  # log scale over at most 3 decades: smaller values (e.g., rounding) get the lightest color
        floor = max(low, high * 1e-3)
        norm = LogNorm(floor, high)
    else:
        floor, norm = 0.0, Normalize(0, high)
    cmap = LinearSegmentedColormap.from_list('ramp', RAMP)
    name = f"{results[0]['mse_factor']:g} × MSE" if args.absolute else 'Relative MSE'
    title = (f"{name} for each layer and bucket of query positions" if args.absolute else
             'Relative MSE (MSE / mean square of the teacher output) for each layer and bucket of query positions')

    ncols = min(4, len(results))
    nrows = math.ceil(len(results) / ncols)
    n_buckets = max(len(r['buckets']) for r in results)
    n_layers = max(len(r['layers']) for r in results)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True,
                             figsize=(ncols * (1.0 + 0.8 * n_buckets) + 1.4, nrows * (1.0 + 0.27 * n_layers)))
    for k, (r, g) in enumerate(zip(results, grids)):
        ax = axes[k // ncols][k % ncols]
        image = ax.imshow([[max(v, floor) for v in row] for row in g], cmap=cmap, norm=norm, aspect='auto')
        for i, row in enumerate(g):
            for j, v in enumerate(row):  # dark ink on light cells, white ink on dark cells
                ax.text(j, i, f'{v:.2g}', ha='center', va='center', fontsize=7,
                        color='white' if norm(max(v, floor)) > 0.5 else TEXT)
        ax.set_xticks(range(len(r['buckets'])))
        ax.set_xticklabels([bucket_name(b) for b in r['buckets']], fontsize=8, color=TEXT_2)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([str(l['layer']) for l in r['layers']], fontsize=8, color=TEXT_2)
        ax.set_xticks([x - 0.5 for x in range(1, len(r['buckets']))], minor=True)  # 2px gaps between cells
        ax.set_yticks([y - 0.5 for y in range(1, len(g))], minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='both', length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(r['label'], color=TEXT, fontsize=9)
        ax.set_xlabel('Query positions', color=TEXT, fontsize=8)
        if k % ncols == 0:
            ax.set_ylabel('Layer', color=TEXT, fontsize=9)
    for k in range(len(results), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle(title, color=TEXT, fontsize=10)
    bar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, extend='min' if low < floor else 'neither')
    bar.set_label(name, color=TEXT, fontsize=8)
    bar.ax.tick_params(colors=TEXT_2, labelsize=8)
    bar.outline.set_visible(False)
    fig.savefig(f'{args.out}.png', dpi=200, facecolor='white')

    # The same values as tables, for the experiment documents
    lines = [title, '']
    for r, g in zip(results, grids):
        names = [bucket_name(b) for b in r['buckets']]
        mean = [b['loss'] if args.absolute else b['relative_mse'] for b in r['buckets']]  # means over the layers
        lines += [f"**{r['label']}**", '', '| Layer | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 1) + '|']
        lines += [f"| {l['layer']} | " + ' | '.join(f'{v:.4g}' for v in row) + ' |' for l, row in zip(r['layers'], g)]
        lines += ['| Mean | ' + ' | '.join(f'{v:.4g}' for v in mean) + ' |', '']
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
    h = commands.add_parser('heatmap')
    h.add_argument('out', help='Output path without extension: writes OUT.png and OUT.md')
    h.add_argument('results', nargs='+', help='JSON files of compute')
    h.add_argument('--absolute', action='store_true', help='mse_factor x MSE instead of the relative MSE')
    r = commands.add_parser('rank')
    r.add_argument('out', help='Output path without extension: writes OUT.md and OUT.png')
    r.add_argument('results', nargs='+', help='JSON files of compute')
    r.add_argument('--accuracy', required=True, help='CSV with the columns label, piqa, arc_easy')
    r.add_argument('--exclude', action='append', default=[], help='Leave out the result with this label')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    {'compute': compute, 'rank': rank, 'heatmap': heatmap}[args.command](args)
