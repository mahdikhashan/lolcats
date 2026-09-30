import ast, statistics, sys
from datasets import load_dataset

line = next(l for l in open(sys.argv[1]) if l.startswith("{'results'"))
results = ast.literal_eval(line)['results']
accs, freqs = [], []
for task, r in results.items():
    if task.startswith('hendrycksTest-'):
        answers = load_dataset('cais/mmlu', task.split('-', 1)[1], split='test')['answer']
        accs.append(r['acc'])
        freqs.append([answers.count(i) / len(answers) for i in range(4)])
print(f'actual macro average: {statistics.mean(accs):.3f}')
for i, letter in enumerate('ABCD'):
    f = [x[i] for x in freqs]
    gap = statistics.mean(abs(a - b) for a, b in zip(accs, f))
    print(f'always "{letter}": would score {statistics.mean(f):.3f}, mean |acc - freq| {gap:.3f}, '
          f'correlation with your per-subject acc {statistics.correlation(accs, f):+.2f}')
