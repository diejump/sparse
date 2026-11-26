"""Serialize the two SPEAR networks and its fixed feature-selection inputs."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def feature_selection(dataset, count):
    epoch = {'Cora': 32, 'Citeseer': 16, 'Pubmed': 128, 'Physics': 128,
             'Flickr': 128, 'ogbn-arxiv': 2048, 'Computers': 128, 'Photo': 128}[dataset]
    path = Path(__file__).resolve().parent / 'save_selected_feature' / dataset / f'val_{epoch}.npy'
    dimensions = np.argsort(np.load(path))[::-1][:count].copy()
    return torch.as_tensor(dimensions), hashlib.sha256(path.read_bytes()).hexdigest()


def generator_metadata(model, args):
    dimensions, digest = feature_selection(args.dataset, args.alpha_int)
    return dict(
        generation_args={key: value for key, value in vars(args).items()
                         if isinstance(value, (str, int, float, bool, type(None)))},
        trigger_generator_state={k: v.detach().cpu().clone() for k, v in model.trojan.state_dict().items()},
        shadow_model_state={k: v.detach().cpu().clone() for k, v in model.shadow_model.state_dict().items()},
        selected_feature_indices=dimensions, feature_selection_sha256=digest,
        alpha_int=args.alpha_int, shadow_hidden=args.hidden)


def load_spear_generator(snapshot, device):
    """Restore the trained embedding network used by inject_trigger as well."""
    from models.backdoor_SPEAR import Backdoor, GraphTrojanNet
    from models.GCN import GCN
    required = {'generation_args', 'shadow_model_state', 'trigger_generator_state',
                'selected_feature_indices', 'alpha_int', 'shadow_hidden'}
    missing = required.difference(snapshot)
    if missing:
        raise ValueError('SPEAR snapshot requires: ' + ', '.join(sorted(missing)))
    args = SimpleNamespace(**snapshot['generation_args'])
    args.alpha_int = int(snapshot['alpha_int'])
    args.hidden = int(snapshot['shadow_hidden'])
    args.dataset = snapshot['dataset']
    dimensions, _ = feature_selection(args.dataset, args.alpha_int)
    if not torch.equal(dimensions, snapshot['selected_feature_indices'].cpu()):
        raise ValueError('SPEAR feature selection differs from the saved snapshot')
    model = Backdoor(args, device)
    model.shadow_model = GCN(args=args, nfeat=snapshot['poison_x'].size(1),
        nhid=args.hidden, nclass=int(snapshot['original_labels'].max()) + 1,
        dropout=0., device=device).to(device)
    model.shadow_model.load_state_dict(snapshot['shadow_model_state'])
    model.shadow_model.eval()
    model.trojan = GraphTrojanNet(device, args.hidden, args.alpha_int, layernum=2).to(device)
    model.trojan.load_state_dict(snapshot['trigger_generator_state'])
    model.trojan.eval()
    return model


def validate_snapshot(snapshot, dataset, num_nodes, generator_path):
    required = {'snapshot_version', 'dataset', 'attack', 'poison_x', 'poison_edge_index',
        'poison_edge_weights', 'poison_labels', 'idx_train', 'idx_attach', 'idx_val',
        'idx_clean_test', 'idx_atk', 'mask_edge_index', 'original_labels',
        'num_original_nodes', 'target_class', 'trigger_generator_state'}
    missing = required.difference(snapshot)
    if missing:
        raise ValueError('SPEAR snapshot requires: ' + ', '.join(sorted(missing)))
    if snapshot['snapshot_version'] != 2 or snapshot['attack'] != 'SPEAR':
        raise ValueError('Expected a version-2 SPEAR snapshot')
    if snapshot['dataset'] != dataset or snapshot['num_original_nodes'] != num_nodes:
        raise ValueError('SPEAR snapshot dataset or node count differs')
    external = torch.load(generator_path, map_location='cpu')
    saved = snapshot['trigger_generator_state']
    if set(external) != set(saved) or any(not torch.equal(external[k].cpu(), saved[k].cpu()) for k in saved):
        raise ValueError('Generator differs from the SPEAR snapshot')
