"""
Layer-wise MSE between the teacher's softmax attention and Lizard attention, from a saved checkpoint
(docs/experiments/xai-layer-wise-mse.md; like Figure 14 of the LoLCATs paper). No training: forward passes only.
-> compute: one stage 1 checkpoint on the stage 1 validation data, written to OUT (JSON)
   python scripts/layer_mse.py compute OUT --model_config M --distill_config D [options]
   - Each attention layer runs in distillation mode (train_attention): it computes the teacher output y_true
     (softmax attention with RoPE) and the Lizard output y_pred from the same q, k, v, before o_proj.
     The next layer sees y_true, so the error of one layer does not reach the next layer.
   - Per layer: MSE(y_pred, y_true) as the stage 1 trainer computes it, the mean square of y_true, their
     ratio (relative MSE) and the MSE of each head. Means over the validation batches.
   - mse_factor x the mean over the layers is the stage 1 validation loss (distill/eval/loss). With the
     default options, it must agree with the loss stored in the checkpoint.
   - --checkpoint: the stage 1 checkpoint. Default: the path distill_llama.py writes for the configs,
     downloaded from HF_REPO if it is not local. `init`: no checkpoint, the Lizard weights at initialization
   - --torch_dtype float32: run a bf16 model config in float32, to compare runs at the same precision
     (the result then does not reproduce the stored loss of a bf16 run exactly)
-> plot: compare RESULT.json files, written to OUT.png and OUT.md (a table of the values)
   python scripts/layer_mse.py plot OUT RESULT.json [RESULT.json ...] [--relative]
   - (a) the MSE of each layer for each result; (b) with 2 or more results, the change of each layer
     against the first result (below 0: a lower MSE than the first result)
"""
import argparse
import json
import os
import subprocess
import sys
from os.path import isfile

import torch

LABEL_HELP = 'Name of this result in the plot and the table (default: the distill config)'
# Categorical slots in fixed order (light mode); the same result keeps its color in both panels
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
TEXT, TEXT_2, GRID = '#0b0b0b', '#52514e', '#e3e2de'


def dump(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)


# --- compute ---

def default_checkpoint(args):
    """The stage 1 checkpoint path of distill_llama.py (see scripts/compare_stages.sh)"""
    run = (f'dl-d={args.distill_config}-m={args.model_config}-f={args.finetune_config}'
           f'-s={args.seed}-se={args.seed}-re={args.replicate}')
    return f'checkpoints/{args.model_config}/{run}_distill.pt'


def download(path, repo_id):
    if isfile(path):  # e.g., downloaded before by compare_stages.sh, or trained on this machine
        return path
    from huggingface_hub import hf_hub_download
    print(f'-> Downloading {path} from {repo_id}')
    return hf_hub_download(repo_id, path, local_dir='.')


def load(args):
    """The model in distillation mode with the stage 1 checkpoint, the configs, the checkpoint check
    and the validation loader. Also used by scripts/position_mse.py"""
    sys.path.insert(0, '.')  # as distill_llama.py, run from the repo root
    sys.path.append('./src')
    from omegaconf import OmegaConf
    from compare_stages import check_checkpoint
    from dataloaders import load_data
    from model.convert_model import toggle_attention
    from model.load_model import load_and_convert_attns
    from model.pretrained import get_pretrained_loader
    from utils.setup import seed_everything

    seed_everything(args.seed)  # the Lizard weights at initialization, if --checkpoint init
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

    checkpoint = args.checkpoint or default_checkpoint(args)
    if checkpoint != 'init':
        checkpoint = download(checkpoint, args.hf_repo)

    attention_type = model_config['attention']['attention_type']
    model_loader = get_pretrained_loader(**model_config.model, huggingface_token=os.environ.get('HF_TOKEN'))
    model = model_loader.load(model_type=attention_type)
    model, _ = load_and_convert_attns(model, model_config, attention_type=attention_type,
                                      checkpoint_path=None if checkpoint == 'init' else checkpoint,
                                      merge_loras=False, train_converted=False, train_attention=True)
    model = toggle_attention(model, train=True)
    model.eval()
    check = None if checkpoint == 'init' else check_checkpoint(model, checkpoint, 'stage1')
    if check is not None:
        print(f"-> stage1 checkpoint: {check['expected_keys']} expected Lizard keys, "
              f"{len(check['missing_keys'])} missing, {len(check['unexpected_keys'])} unexpected, "
              f"{len(check['values_not_loaded'])} not loaded, step {check['step']}, losses {check['losses']}")

    dataloaders = load_data(distill_config.dataset, distill_config.dataloader)
    return model, model_config, distill_config, check, dataloaders[distill_config.trainer.val_split]


def compute(args):
    model, model_config, distill_config, check, loader = load(args)
    criterion = torch.nn.MSELoss(reduction='mean')  # as in src/trainer/distill_attention_xent_mse.py
    sums, batches, tokens = None, 0, 0
    with torch.no_grad():
        for ix, data in enumerate(loader):
            if args.max_batches is not None and ix == args.max_batches:
                break
            inputs = {k: v.to(model.device) for k, v in data.items() if k != 'labels'}
            attentions = model(**inputs, output_attentions=True, use_cache=False).get('attentions')
            values = []
            for layer, attns in enumerate(attentions):  # ((None, None), (y_pred, y_true)), shape (b, h, l, d)
                assert attns is not None, f'Layer {layer} is not a Lizard layer (softmax_attentions)'
                y_pred, y_true = attns[1]
                error = (y_pred.float() - y_true.float()).pow(2)
                values.append([criterion(y_pred, y_true).item(), y_true.float().pow(2).mean().item()]
                              + error.mean(dim=(0, 2, 3)).tolist())
            sums = values if sums is None else [[s + v for s, v in zip(a, b)] for a, b in zip(sums, values)]
            batches += 1
            tokens += inputs['input_ids'].numel()
            print(f'-> Batch {ix + 1}: {distill_config.trainer.mse_factor} x mean layer MSE = '
                  f'{distill_config.trainer.mse_factor * sum(v[0] for v in values) / len(values):.4f}')
    assert batches > 0, 'No validation batches'

    layers = []
    for i, s in enumerate(sums):
        mse, target_ms, *heads = [x / batches for x in s]
        layers.append({'layer': i, 'mse': mse, 'target_mean_square': target_ms,
                       'relative_mse': mse / target_ms, 'mse_per_head': heads})
    mse_factor = float(distill_config.trainer.mse_factor)
    loss = mse_factor * sum(l['mse'] for l in layers) / len(layers)
    stored = (check or {}).get('losses', {}).get(distill_config.trainer.metric_for_best_model)
    result = {
        'label': args.label or args.distill_config,
        'model_config': args.model_config,
        'distill_config': args.distill_config,
        'torch_dtype': str(model_config.model.torch_dtype),
        'checkpoint': check or {'path': 'init', 'seed': args.seed},
        'data': {'split': distill_config.trainer.val_split, 'batches': batches, 'tokens': tokens,
                 'all_batches': args.max_batches is None},
        'mse_factor': mse_factor,
        'loss': loss,  # same scale and data as distill/eval/loss
        'stored_loss': stored,
        'layers': layers,
        'env': {'torch': torch.__version__, 'device': str(model.device),
                'lolcats_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                                 text=True).stdout.strip() or None},
    }
    dump(result, args.out)
    print(f'\n{"Layer":>5} {"MSE":>10} {"relative":>9}')
    for l in layers:
        print(f"{l['layer']:>5} {l['mse']:>10.3e} {l['relative_mse']:>9.4f}")
    print(f'-> {mse_factor:g} x mean layer MSE = {loss:.4f} over {batches} batches ({tokens} tokens); '
          f'stored loss in the checkpoint: {stored}')
    if stored is not None and args.max_batches is None:
        print(f'-> Difference to the stored loss: {100 * (loss - stored) / stored:+.3f}%')
    print(f'-> Wrote {args.out}')


# --- plot ---

def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = []
    for path in args.results:
        with open(path) as f:
            results.append(json.load(f))
    assert len(results) <= len(COLORS), f'At most {len(COLORS)} results in one plot (got {len(results)})'
    key = 'relative_mse' if args.relative else 'mse'
    scale = lambda r: 1.0 if args.relative else r['mse_factor']
    value = lambda r: [scale(r) * l[key] for l in r['layers']]
    name = 'Relative MSE (MSE / mean square of the teacher output)' if args.relative else \
        f"{results[0]['mse_factor']:g} × MSE (the scale of distill/eval/loss)"
    n_layers = max(len(r['layers']) for r in results)

    panels = 2 if len(results) > 1 else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.8 * panels, 4.4), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(TEXT_2)
        ax.tick_params(colors=TEXT_2, labelsize=9)
        ax.set_xticks(range(n_layers))
        ax.set_xlabel('Layer', color=TEXT)

    ax = axes[0]
    for i, r in enumerate(results):
        layers = [l['layer'] for l in r['layers']]
        label = r['label'] if args.relative else f"{r['label']} (mean {r['loss']:.4f})"
        ax.plot(layers, value(r), color=COLORS[i], linewidth=2, marker='o', markersize=5,
                markeredgecolor='white', markeredgewidth=1, label=label)
    ax.set_ylabel(name, color=TEXT)
    ax.set_title('(a) Layer-wise MSE', color=TEXT, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)

    if panels == 2:
        ax = axes[1]
        ref = value(results[0])
        others = results[1:]
        width = 0.8 / len(others)
        for j, r in enumerate(others):
            change = [v - b for v, b in zip(value(r), ref)]
            x = [l + (j - (len(others) - 1) / 2) * width for l in range(len(change))]
            ax.bar(x, change, width=width * 0.9, color=COLORS[j + 1], label=r['label'])
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_ylabel(f"Change against {results[0]['label']}", color=TEXT)
        ax.set_title('(b) Layer-wise MSE change (below 0: lower MSE)', color=TEXT, fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(f'{args.out}.png', dpi=200, facecolor='white')

    # The same values as a table, for the experiment documents
    rows = ['| Layer | ' + ' | '.join(r['label'] for r in results) + ' |', '|---' * (len(results) + 1) + '|']
    for i in range(n_layers):
        rows.append(f'| {i} | ' + ' | '.join(f'{value(r)[i]:.4g}' if i < len(r['layers']) else '–'
                                             for r in results) + ' |')
    if not args.relative:
        rows.append('| Mean (the loss) | ' + ' | '.join(f"{r['loss']:.4f}" for r in results) + ' |')
        rows.append('| Stored loss | ' + ' | '.join('–' if r['stored_loss'] is None else f"{r['stored_loss']:.4f}"
                                                   for r in results) + ' |')
    with open(f'{args.out}.md', 'w') as f:
        f.write(f'{name}, per layer\n\n' + '\n'.join(rows) + '\n')
    print('\n'.join(rows))
    print(f'-> Wrote {args.out}.png and {args.out}.md')


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('compute')
    c.add_argument('out', help='Output JSON file')
    c.add_argument('--model_config', required=True)
    c.add_argument('--distill_config', required=True)
    c.add_argument('--finetune_config', default='finetune_lora_qkvo_alpaca_clean_1b',
                   help='Only for the default checkpoint name')
    c.add_argument('--checkpoint', default=None, help='Stage 1 checkpoint, or `init` (see the top of this file)')
    c.add_argument('--hf_repo', default=os.environ.get('HF_REPO', 'nanoman1/lolcats-lizard-llama-3.2-1b'))
    c.add_argument('--label', default=None, help=LABEL_HELP)
    c.add_argument('--torch_dtype', default=None, help='Replaces the model config dtype, e.g. float32')
    c.add_argument('--max_batches', type=int, default=None, help='Default: all validation batches')
    c.add_argument('--cache_dir', default=None, help="Replaces the model config's cache_dir")
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--replicate', type=int, default=0)
    p = commands.add_parser('plot')
    p.add_argument('out', help='Output path without extension: writes OUT.png and OUT.md')
    p.add_argument('results', nargs='+', help='JSON files of compute; the first is the reference of panel (b)')
    p.add_argument('--relative', action='store_true', help='Plot the relative MSE instead of the MSE')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    compute(args) if args.command == 'compute' else plot(args)
