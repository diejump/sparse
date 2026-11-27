"""Count cached selections using labels only; no model imports or training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run_experiments import ATTACK_SETTINGS


def retained(candidates, labels, target):
    candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
    if candidates.size and (candidates.min() < 0 or candidates.max() >= len(labels)):
        raise ValueError('Candidate index outside the label vector')
    result = candidates[labels[candidates] != target]
    return dict(candidate_count=int(len(candidates)), retained_count=int(len(result)),
                unique_retained_count=int(len(np.unique(result))),
                removed_target_count=int(len(candidates) - len(result)),
                retained_nodes=result.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    rows, sources = [], []
    for dataset, desired in [('Cora', 40), ('Citeseer', 40), ('Pubmed', 160), ('Physics', 160)]:
        path = args.data_root / dataset / ('Physics/processed/data.pt' if dataset == 'Physics'
                                         else 'processed/data.pt')
        loaded = torch.load(path, map_location='cpu')
        # PyG stores either Data or its plain attribute dictionary in position zero.
        data = loaded[0] if isinstance(loaded, tuple) else loaded
        y = data['y'] if isinstance(data, dict) else data.y
        y = y.reshape(-1).cpu().numpy()
        sources.append(dict(dataset=dataset, path=str(path), nodes=len(y),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        permutation = np.random.RandomState(10).permutation(len(y))
        excluded = set(permutation[:int(.2 * len(y))].tolist())
        test_start = int(.2 * len(y)) + int(.1 * len(y))
        excluded.update(permutation[test_start:test_start + int(.2 * len(y))].tolist())
        unlabeled = np.asarray([i for i in range(len(y)) if i not in excluded], dtype=np.int64)
        for algorithm in ATTACK_SETTINGS:
            cfg = ATTACK_SETTINGS[algorithm][dataset]
            budget, target, seed = cfg['vs_number'], cfg['target_class'], cfg['seed']
            row = dict(dataset=dataset, algorithm=algorithm, desired=desired, vs_number=budget,
                       seed=seed, target_class=target, cache_sources=[])
            if algorithm in ('GTA', 'DPGBA'):
                choice = np.arange(len(unlabeled))
                np.random.RandomState(seed).shuffle(choice)
                candidates = unlabeled[choice[:min(len(unlabeled), budget)]]
                row['selection_source'] = 'fixed split + seeded NumPy shuffle'
            elif algorithm == 'SPEAR':
                cache = project / 'sorted_samples' / dataset / 'sorted_samples.pt'
                candidates = torch.load(cache, map_location='cpu').reshape(-1).numpy()[:budget]
                row['selection_source'] = str(cache)
                row['cache_sources'].append(dict(path=str(cache), sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
            elif dataset == 'Pubmed':
                # The source always fits an encoder before consulting class caches.
                # With all cached non-target class lists populated, the returned
                # prefixes have this exact count independently of fresh scores.
                pieces = []
                per_class = int(budget / (int(y.max()) + 1 - 1))
                for label in range(int(y.max()) + 1):
                    if label == target:
                        continue
                    size = budget - sum(map(len, pieces)) if label == int(y.max()) else per_class
                    cache = project / 'selected_nodes' / dataset / 'Seperate' / f'seed{seed}' / f'class_{label}.txt'
                    pieces.append(np.loadtxt(cache, delimiter=',').astype(np.int64).reshape(-1)[:size])
                    row['cache_sources'].append(dict(path=str(cache), sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
                candidates = np.concatenate(pieces)
                row['selection_source'] = 'cached class prefixes; encoder class-presence condition'
                row['qualification'] = 'Conditional on the freshly fitted encoder predicting both non-target classes'
            else:
                cache = project / 'selected_nodes' / dataset / 'Overall' / f'seed{seed}' / 'nodes.txt'
                candidates = np.loadtxt(cache, delimiter=',').astype(np.int64).reshape(-1)[:budget]
                row['selection_source'] = str(cache)
                row['cache_sources'].append(dict(path=str(cache), sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
            row.update(retained(candidates, y, target))
            row['matches_desired'] = row['unique_retained_count'] == desired
            row['outside_unlabeled'] = int(sum(int(i) in excluded for i in candidates))
            rows.append(row)
            print(dataset, algorithm, 'budget', budget, 'retained', row['retained_count'],
                  'desired', desired, row.get('qualification', ''), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(sources=sources, rows=rows), ensure_ascii=False, indent=2),
                           encoding='utf-8')


if __name__ == '__main__':
    main()
