"""Evaluate defenses against a paired, fixed SPEAR graph and generator."""
from deterministic_runtime import prepare_environment, configure_determinism, seed_classifier
prepare_environment()
import argparse
import hashlib
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid, Flickr, Coauthor
import torch_geometric.transforms as T

from experiment_defenses import add_arguments, run_comparison
from help_funcs import prune_unrelated_edge
from models.construct import model_construct
from spear_snapshot import validate_snapshot, load_spear_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--classifier_seed', type=int)
    parser.add_argument('--dataset', default='Cora',
        choices=['Cora', 'Citeseer', 'Pubmed', 'Physics', 'Flickr', 'ogbn-arxiv'])
    parser.add_argument('--model', default='GCN')
    parser.add_argument('--test_model', choices=['GCN', 'GAT', 'GraphSage'], default='GCN')
    parser.add_argument('--hidden', type=int, default=80)
    parser.add_argument('--train_lr', type=float, default=.01)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--target_class', type=int, default=0)
    parser.add_argument('--defense_mode', default='none',
        choices=['none', 'losplit', 'prune', 'abp', 'abp_sparse'])
    parser.add_argument('--skip_rigbd', action='store_true')
    parser.add_argument('--rigbd_k', type=int, default=20)
    parser.add_argument('--rigbd_b', type=float, default=.5)
    parser.add_argument('--prune_thr', type=float, default=.1)
    parser.add_argument('--trigger_generator_address', required=True)
    parser.add_argument('--pre_train_param', required=True)
    parser.add_argument('--model_save_path', default='')
    add_arguments(parser)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device_id}' if args.cuda else 'cpu')
    if args.deterministic:
        configure_determinism(args.seed, args.cuda)
    seed_classifier(args.seed, args.cuda)
    if args.rigbd_k < 1 or not 0 <= args.rigbd_b <= 1:
        parser.error('rigbd_k must be positive and rigbd_b must lie in [0,1]')
    if args.dataset in ('Cora', 'Citeseer', 'Pubmed'):
        dataset = Planetoid(root='./data/', name=args.dataset, transform=T.NormalizeFeatures())
    elif args.dataset == 'Physics':
        dataset = Coauthor(root='./data/Physics', name='Physics')
    elif args.dataset == 'Flickr':
        dataset = Flickr(root='./data/Flickr/', transform=T.NormalizeFeatures())
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
    data = dataset[0].to(device)
    snapshot = torch.load(args.pre_train_param, map_location=device)
    validate_snapshot(snapshot, args.dataset, data.x.size(0), args.trigger_generator_address)
    data.y = snapshot['original_labels']
    args.target_class = int(snapshot['target_class'])
    generator = load_spear_generator(snapshot, device)
    print(args, flush=True)
    print('SNAPSHOT_SHA256:', hashlib.sha256(Path(args.pre_train_param).read_bytes()).hexdigest(), flush=True)
    print('GENERATOR_MATCHES_SNAPSHOT: True', flush=True)
    print('SPEAR_SHADOW_RESTORED: True', flush=True)
    if args.model_save_path:
        Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    run_comparison(args, data, device, snapshot, generator, model_construct, prune_unrelated_edge)


if __name__ == '__main__':
    main()