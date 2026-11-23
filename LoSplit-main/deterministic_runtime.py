"""Opt-in runtime configuration; no model, loss, optimizer or sampling replacement."""
import os
import sys


def prepare_environment():
    if '--deterministic' in sys.argv:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def seed_classifier(seed, use_cuda):
    """Start an independent classifier RNG stream immediately before construction.

    This resets RNGs only; deterministic algorithm/environment setup stays at startup.
    PYTHONHASHSEED is a process-start setting and need not equal this phase's seed.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    print('CLASSIFIER_SEED:', seed, flush=True)


def configure_determinism(seed, use_cuda):
    import json
    import random
    import numpy as np
    import torch
    if os.environ.get('PYTHONHASHSEED') != str(seed):
        raise ValueError('Launch Python with PYTHONHASHSEED={} for strict mode'.format(seed))
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('DETERMINISTIC_RUNTIME:', json.dumps({
        'enabled': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'device': 'cuda' if use_cuda else 'cpu', 'seed': seed,
        'PYTHONHASHSEED': os.environ['PYTHONHASHSEED'],
        'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'torch': torch.__version__, 'cuda_version': torch.version.cuda,
        'threads': torch.get_num_threads(), 'tf32': False}), flush=True)
