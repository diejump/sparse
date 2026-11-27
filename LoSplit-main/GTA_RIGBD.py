#!/usr/bin/env python
# coding: utf-8

# In[1]: 


import os
from deterministic_runtime import prepare_environment
prepare_environment()
import imp
import time
import argparse
from run_experiments import ALGORITHM_DEFAULTS, PRUNE_SETTINGS, configure_algorithm_parser
from copy import deepcopy
from unittest import result
import numpy as np
import torch
import seaborn as sns
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI
from help_funcs import reconstruct_prune_unrelated_edge


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge,prune_unrelated_edge_isolated
import scipy.sparse as sp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import gaussian_kde
from scipy.stats import norm
from torch.distributions.bernoulli import Bernoulli
from collections import Counter

from sklearn.metrics import pairwise_distances
from ogb.nodeproppred import PygNodePropPredDataset

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams   # ✅ 这样才能用 rcParams
rcParams['font.sans-serif'] = ['Songti SC']   # 支持中文
rcParams['font.family'] = 'sans-serif'
font_cn = FontProperties(family="Songti SC", size=18)      # 中文宋体
font_en = FontProperties(family="Times New Roman", size=20) # 英文 + 数字
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 25  

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--deterministic', action='store_true', help='Strict deterministic runtime for audited Cora/GCN settings')
parser.add_argument('--rigbd_k', type=int, default=20)
parser.add_argument('--rigbd_b', type=float, default=0.5, help='Random edge drop ratio')
parser.add_argument('--skip_rigbd', action='store_true', help='Evaluate only the selected graph defense')
parser.add_argument('--model_save_path', type=str, default='')
parser.add_argument('--abp_min_branch', type=int, default=2)
parser.add_argument('--abp_recovery_epochs', type=int, default=400)
parser.add_argument('--abp_negative_weight', type=float, default=3.0)
parser.add_argument('--abp_teacher_hidden', type=int, default=32)
parser.add_argument('--abp_teacher_epochs', type=int, default=200)
parser.add_argument('--abp_confidence', type=float, default=0.8)
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=ALGORITHM_DEFAULTS['GTA']['seed'], help='Random seed.')
parser.add_argument('--classifier_seed', type=int, default=None,
                    help='Independent evaluation classifier seed; defaults to --seed')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp', 'Physics'])
parser.add_argument('--train_lr', type=float, default=ALGORITHM_DEFAULTS['GTA']['train_lr'],
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=ALGORITHM_DEFAULTS['GTA']['weight_decay'],
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=ALGORITHM_DEFAULTS['GTA']['hidden'],
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=ALGORITHM_DEFAULTS['GTA']['thrd'])

parser.add_argument('--target_class', type=int, default=ALGORITHM_DEFAULTS['GTA']['target_class'])

parser.add_argument('--dropout', type=float, default=ALGORITHM_DEFAULTS['GTA']['dropout'],
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=ALGORITHM_DEFAULTS['GTA']['epochs'], help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=ALGORITHM_DEFAULTS['GTA']['trojan_epochs'], help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=ALGORITHM_DEFAULTS['GTA']['inner'], help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=ALGORITHM_DEFAULTS['GTA']['lr'],
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=ALGORITHM_DEFAULTS['GTA']['trigger_size'],
                    help='tirgger_size')
parser.add_argument('--vs_number', type=int, default=ALGORITHM_DEFAULTS['GTA']['vs_number'],
                    help="ratio of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="none",
                    choices=['prune', 'isolate', 'none', 'reconstruct', 'abp','losplit','abp_sparse','abp_dual','abp_assisted'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=PRUNE_SETTINGS['GTA'],
                    help="Threshold of prunning edges")
parser.add_argument('--homo_loss_weight', type=float, default=ALGORITHM_DEFAULTS['GTA']['homo_loss_weight'],
                    help="Weight of optimize similarity loss")


# attack setting
parser.add_argument('--dis_weight', type=float, default=ALGORITHM_DEFAULTS['GTA']['dis_weight'],
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default=ALGORITHM_DEFAULTS['GTA']['selection_method'],
                    choices=['loss','conf','cluster','none','cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GIN','DGI', 'GNNGuard','RobustGCN', 'ABL'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='overall',
                    choices=['overall','1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
parser.add_argument('--rec_epochs', type=int,  default=ALGORITHM_DEFAULTS['GTA']['rec_epochs'],
                    help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trigger_generator_address', type=str, default='./weights/GTA/Cora/GTA_Cora_weights.pth')
parser.add_argument('--pre_train_param', type=str, default='./weights/GTA/Cora/GTA_Cora_RIGBD.pt')
from experiment_defenses import add_arguments, run_comparison
add_arguments(parser)
configure_algorithm_parser(parser, 'GTA')
args = parser.parse_args()
if args.defense_mode in ('abp', 'abp_dual', 'abp_assisted'):
    args.skip_rigbd = True
if args.rigbd_k < 1 or not 0 <= args.rigbd_b <= 1:
    parser.error('rigbd_k must be positive and rigbd_b must be in [0,1]')
# CPU请求或显式隐藏GPU时，直接选择CPU，避免初始化CUDA运行时。
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
cuda_disabled = args.no_cuda or (
    cuda_visible_devices is not None and cuda_visible_devices.strip() in ('', '-1')
)
args.cuda = not cuda_disabled and torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.device_id) if args.cuda else 'cpu')
if args.deterministic:
    from deterministic_runtime import configure_determinism
    configure_determinism(args.seed, args.cuda)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)
#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif(args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                    transform=transform)
elif(args.dataset == 'Reddit2'):
    dataset = Reddit2(root='./data/Reddit2/', \
                    transform=transform)
elif(args.dataset == 'Physics'):
    from torch_geometric.datasets import Coauthor
    dataset = Coauthor(root='./data/Physics', name='Physics')
elif(args.dataset == 'ogbn-arxiv'):
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif(args.dataset=='Physics'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


# In[9]:
from models.GTA import Backdoor, GraphTrojanNet
import heuristic_selection as hs

# from kmeans_pytorch import kmeans, kmeans_predict


# ===== LOAD_FIXED_POISON_SNAPSHOT: 直接读取攻击入口保存的固定样本 =====
from models.construct import model_construct
poison_snapshot = torch.load(args.pre_train_param, map_location=device)
required_snapshot_keys = {'snapshot_version', 'dataset', 'attack', 'poison_x',
    'poison_edge_index', 'poison_edge_weights', 'poison_labels', 'idx_train',
    'idx_attach', 'idx_val', 'idx_clean_test', 'idx_atk', 'mask_edge_index',
    'original_labels', 'num_original_nodes', 'target_class', 'trigger_size', 'trigger_threshold',
    'trigger_generator_state'}
missing_snapshot_keys = required_snapshot_keys.difference(poison_snapshot)
if missing_snapshot_keys:
    raise ValueError('投毒缓存缺少固定划分/生成器信息：' + ', '.join(sorted(missing_snapshot_keys))
                     + '。请使用新版攻击快照；旧文件需从原运行记录补齐元数据。')
if poison_snapshot['snapshot_version'] != 2:
    raise ValueError('不支持的投毒快照版本。')
if poison_snapshot['dataset'] != args.dataset or poison_snapshot['attack'] != 'GTA':
    raise ValueError('投毒快照的数据集或攻击类型与当前入口不一致。')
if poison_snapshot['num_original_nodes'] != data.x.size(0):
    raise ValueError('投毒快照与当前数据集的原节点数不一致。')
poison_x = poison_snapshot['poison_x']
poison_edge_index = poison_snapshot['poison_edge_index']
poison_edge_weights = poison_snapshot['poison_edge_weights']
poison_labels = poison_snapshot['poison_labels']
idx_train = poison_snapshot['idx_train']
idx_attach = poison_snapshot['idx_attach']
idx_val = poison_snapshot['idx_val']
idx_clean_test = poison_snapshot['idx_clean_test']
idx_atk = poison_snapshot['idx_atk']
mask_edge_index = poison_snapshot['mask_edge_index']
data.y = poison_snapshot['original_labels']
args.target_class = int(poison_snapshot['target_class'])
args.trigger_size = int(poison_snapshot['trigger_size'])
args.thrd = float(poison_snapshot['trigger_threshold'])
known_nodes = torch.cat([idx_train, idx_attach]).to(device)
# 只恢复生成器，用于原评估代码给测试节点添加触发器；不调用fit/get_poisoned。
model = Backdoor(args, device)
model.trojan = GraphTrojanNet(device, poison_x.size(1), args.trigger_size, layernum=2).to(device)
model.trojan.load_state_dict(poison_snapshot['trigger_generator_state'])
model.trojan.eval()
print('Loaded fixed poison snapshot:', args.pre_train_param)
import hashlib
with open(args.pre_train_param, 'rb') as snapshot_file:
    print('SNAPSHOT_SHA256:', hashlib.sha256(snapshot_file.read()).hexdigest())
external_generator_state = torch.load(args.trigger_generator_address, map_location='cpu')
assert set(external_generator_state) == set(poison_snapshot['trigger_generator_state']), 'Generator keys differ from snapshot'
assert all(torch.equal(external_generator_state[key].cpu(), poison_snapshot['trigger_generator_state'][key].cpu()) for key in external_generator_state), 'Generator differs from fixed snapshot'
print('GENERATOR_MATCHES_SNAPSHOT: True')

# ===== 原选点及训练图重建代码（已停用） =====
# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
# unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
# size = args.vs_number #int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
# print("#Attach Nodes:{}".format(size))
# here is randomly select poison nodes from unlabeled nodes
# from models.construct import model_construct
# if(args.selection_method == 'none'):
#     idx_attach = hs.obtain_attach_nodes(args,unlabeled_idx,size)
# elif(args.selection_method == 'cluster'):
#     idx_attach = hs.cluster_distance_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
#     idx_attach = torch.LongTensor(idx_attach).to(device)
# elif(args.selection_method == 'cluster_degree'):
#     idx_attach = hs.cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
#     idx_attach = torch.LongTensor(idx_attach).to(device)
# mask = data.y[idx_attach] != args.target_class
# mask = mask.to(device)
# idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
# print(len(idx_attach))

# known_nodes = torch.cat([idx_train,idx_attach]).to(device)


# model = Backdoor(args,device)
# model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
# torch.save(model.trojan.state_dict(), args.trigger_generator_address)

# model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx, args.trigger_generator_address, True)
# poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()


# ===== 固定快照加载结束，继续作者原防御代码 =====
if args.defense_mode in ('none', 'losplit', 'abp', 'abp_sparse', 'prune'):
    run_comparison(args, data, device, poison_snapshot, model, model_construct, prune_unrelated_edge)
    raise SystemExit(0)
if args.defense_mode == 'abp':
    from models.ABP import filter_edges
    # Keep the complete observed graph for the two separate evaluation projections.
    abp_raw_train_edges = poison_edge_index
    abp_raw_train_weights = poison_edge_weights
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
    poison_edge_index, poison_edge_weights = filter_edges(
        abp_raw_train_edges, abp_raw_train_weights, poison_x.size(0), bkd_tn_nodes)
    print('ABP_TRAIN_ARCS:', abp_raw_train_edges.size(1), '->', poison_edge_index.size(1))
elif(args.defense_mode == 'prune'):
    poison_edge_index,poison_edge_weights = prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'reconstruct'):
    poison_edge_index,poison_edge_weights = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,data.x,data.edge_index,device, idx_attach, large_graph=True)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
elif(args.defense_mode == 'isolate'):
    poison_edge_index,poison_edge_weights,rel_nodes = prune_unrelated_edge_isolated(args,poison_edge_index,poison_edge_weights,poison_x,device,large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}"\
    .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach)))
    #%%


        

# ===== 原独立无防御基线（已停用） =====
# test_model = model_construct(args,args.test_model,data,device).to(device) 

# if test_model == 'ABL':
#     test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, num_attach=len(idx_attach))
# else:
#     test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
# test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
# test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
# test_model.fit(data.x, data.edge_index, data.edge_weight, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False) 


# output = test_model(poison_x, poison_edge_index, poison_edge_weights)
# induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
# induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

# train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
# print("target class rate on Vs: {:.4f}".format(train_attach_rate))
# clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

# print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


# induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
# induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
# output = test_model(induct_x,induct_edge_index,induct_edge_weights)
# train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
# print("target class rate on Vs: {:.4f}".format(train_attach_rate))
# if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
#     induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)

# output = test_model(induct_x,induct_edge_index,induct_edge_weights)

# flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
# flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()

# print("****After GTA Attack****")
# print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
# print("Clean Accuracy: {:.6f}".format(clean_acc))
# induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
# output = output.cpu()



# ===== 保留图处理防御所需的分类器训练与评估 =====
if args.defense_mode != "none" or args.skip_rigbd:
    from deterministic_runtime import seed_classifier
    classifier_seed = args.seed if args.classifier_seed is None else args.classifier_seed
    if args.defense_mode in ('abp_dual', 'abp_assisted'):
        from models.ABP_Learning import ABPLearning
        seed_classifier(classifier_seed, args.cuda)
        test_model = ABPLearning(args, args.test_model, data, device, tuple_output=False).to(device)
    else:
        seed_classifier(classifier_seed, args.cuda)
        test_model = model_construct(args,args.test_model,data,device).to(device) 

    if test_model == 'ABL':
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False, num_attach=len(idx_attach))
    else:
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    # test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
    # test_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
    # test_model.fit(data.x, data.edge_index, data.edge_weight, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False) 

    test_model.eval()
    output = test_model(poison_x, poison_edge_index, poison_edge_weights)
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])

    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    print("target class rate on Vs: {:.4f}".format(train_attach_rate))
    if args.defense_mode == 'abp':
        abp_eval_raw_edges = torch.cat([abp_raw_train_edges, mask_edge_index], dim=1)
        abp_eval_raw_weights = torch.cat([abp_raw_train_weights, poison_x.new_ones(mask_edge_index.size(1))])
        induct_edge_index, induct_edge_weights = filter_edges(
            abp_eval_raw_edges, abp_eval_raw_weights, poison_x.size(0), bkd_tn_nodes)
    clean_acc = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

    print("accuracy on clean test nodes: {:.4f}".format(clean_acc))


    if args.defense_mode == 'abp':
        # Inject on the original observed evaluation graph, then project the full triggered graph.
        induct_edge_index, induct_edge_weights = abp_eval_raw_edges, abp_eval_raw_weights
    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
    # output = test_model(induct_x,induct_edge_index,induct_edge_weights)
    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    print("target class rate on Vs: {:.4f}".format(train_attach_rate))
    if(args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
        induct_edge_index,induct_edge_weights = prune_unrelated_edge(args,induct_edge_index,induct_edge_weights,induct_x,device)
    elif args.defense_mode == 'abp':
        induct_edge_index, induct_edge_weights = filter_edges(
            induct_edge_index, induct_edge_weights, induct_x.size(0), bkd_tn_nodes)

    output = test_model(induct_x,induct_edge_index,induct_edge_weights)

    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()

    print("****After GTA Attack****")
    print("ASR: {:.6f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    print("Clean Accuracy: {:.6f}".format(clean_acc))
    induct_x, induct_edge_index,induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(),induct_edge_weights.cpu()
    output = output.cpu()




if args.skip_rigbd:
    if args.model_save_path:
        torch.save(test_model.state_dict(), args.model_save_path)
        if hasattr(test_model, 'defense_metadata'):
            import json
            with open(args.model_save_path + '.meta.json', 'w', encoding='utf-8') as metadata_file:
                json.dump(test_model.defense_metadata, metadata_file, ensure_ascii=False, indent=2)
    print('FINAL_DEFENSE_RESULT method={} CA={:.9f} ASR={:.9f}'.format(args.defense_mode, float(clean_acc), float(flip_asr)))
    raise SystemExit(0)

# RIGBD Defense
test_model = model_construct(args,args.test_model,data,device,add_self_loops=False).to(device) 
test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)
test_model.eval()
clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
output = test_model(poison_x, poison_edge_index, poison_edge_weights)
ori_predict = torch.exp(output[known_nodes])
# print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

K_list = [args.rigbd_k]
drop_ratio_list = [args.rigbd_b]
results = {}

for drop_ratio in drop_ratio_list:
    for K in K_list:
        print(f"Running for drop_ratio={drop_ratio}, K={K}")

        # 重新定义 sample_noise_all，传入当前 drop_ratio
        def sample_noise_all(edge_index, edge_weight, device, drop_ratio=drop_ratio):
            edge_index = edge_index.to(device)
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=device)
            else:
                edge_weight = edge_weight.to(device)

            drop_mask = Bernoulli(1 - drop_ratio).sample(edge_weight.size()).bool()
            noisy_edge_index = edge_index[:, drop_mask]
            noisy_edge_weight = edge_weight[drop_mask]

            node_degrees = torch.zeros(edge_index.max() + 1, device=device)
            node_degrees.index_add_(0, noisy_edge_index[0], torch.ones(noisy_edge_index.size(1), device=device))

            isolated_nodes = node_degrees == 0
            if isolated_nodes.any():
                potential_restore_edges = isolated_nodes[edge_index[0]]
                restore_edges = edge_index[:, potential_restore_edges]
                noisy_edge_index = torch.cat([noisy_edge_index, restore_edges], dim=1)
                restored_weights = torch.ones(restore_edges.size(1), device=device)
                noisy_edge_weight = torch.cat([noisy_edge_weight, restored_weights], dim=0)

            return noisy_edge_index, noisy_edge_weight
        
        def find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, target_class):
            # Get the specific list to iterate through
            labels_list = poison_labels[bkd_tn_nodes[index_of_less_robust]]

            # Iterate through the list with index
            for i in range(len(labels_list) - 1):  # -1 to avoid index out of range
                if labels_list[i] != target_class and labels_list[i + 1] != target_class:
                    return i - 1

        # 清空历史predictions
        predictions = []
        
        # 用当前K跑
        for i in range(K):
            test_model.eval()
            noisy_poison_edge_index, noisy_poison_edge_weights = sample_noise_all(poison_edge_index, poison_edge_weights, device)
            output = test_model(poison_x, noisy_poison_edge_index, noisy_poison_edge_weights)
            predictions.append(torch.exp(output[known_nodes]))

        epsilon = 1e-8
        deviations = []
        for sub_pred in predictions:
            sub_pred += epsilon
            deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
            deviations.append(deviation)

        summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
        for deviation in deviations:
            summed_deviations += deviation
        
        PV = torch.mean(summed_deviations, dim=-1)
        PV = PV.detach().cpu().numpy()
        plt.figure(figsize=(7,5))

        # 干净节点
        clean_nodes = PV[:-len(idx_attach)]
        n_clean = len(clean_nodes)

        # 随机打乱索引并分成两半
        indices = np.random.permutation(n_clean)
        half = n_clean // 2
        idx_blue = indices[:half]
        idx_green = indices[half:]

        plt.hist(clean_nodes[idx_blue], bins=30, color='blue', alpha=0.6, label="干净节点 (类0)")
        plt.hist(clean_nodes[idx_green], bins=30, color='green', alpha=0.6, label="干净节点 (类1)")

        # 受感染节点
        plt.hist(PV[-len(idx_attach):], bins=30, color='crimson', alpha=0.8, label="受感染节点")

        plt.xlabel("删边前后预测值差异", fontsize=28)
        plt.ylabel("节点数量", fontsize=28)
        plt.legend(fontsize=18)
        plt.tight_layout()
        plt.savefig("prediction_variance.pdf", format="pdf")
        plt.show()

        index_of_less_robust = torch.sort(torch.mean(summed_deviations, dim=-1), descending=True)[1]
        result_index = len(idx_attach)

        idx_poison_found = known_nodes[index_of_less_robust][:result_index]
        idx_clean_found = known_nodes[index_of_less_robust[result_index:]]
        poison_common = set(idx_poison_found.tolist()) & set(idx_attach.tolist())
        clean_common = set(idx_clean_found.tolist()) & set(idx_train.tolist())


        TP = len(poison_common)   # 正确检测出的中毒节点
        FP = len(idx_poison_found) - TP  # 被误判为中毒节点的干净节点
        FN = len(idx_attach) - TP  # 没有被检测出的中毒节点
        TN = len(clean_common) # 正确识别的干净节点

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        fpr = FP / (FP + TN + 1e-8)


        ####### 训练防御后模型 #######
        test_model = model_construct(args, args.test_model, data, device).to(device)
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=400, verbose=False, finetune1=True, attach=known_nodes[index_of_less_robust][:result_index])

        induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
        induct_edge_weights = torch.cat([poison_edge_weights, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])

        clean_acc = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

        induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(idx_atk, poison_x, induct_edge_index, induct_edge_weights, device)
        induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()

        output = test_model(induct_x, induct_edge_index, induct_edge_weights)
        flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
        flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
        asr = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean().item()
        ca = test_model.test(induct_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

        results[(drop_ratio, K)] = {
            'ASR': flip_asr,
            'CleanACC': clean_acc,
            'Precision': precision,
            'Recall': recall,
            'FPR': fpr,
        }

        for key, value in results.items():
            print(f"drop_ratio={key[0]}, K={key[1]} -> "
                f"ASR: {value['ASR']:.6f}, "
                f"CleanAcc: {value['CleanACC']:.6f}, "
                f"Precision: {value['Precision']:.6f}, "
                f"Recall: {value['Recall']:.6f}, "
                f"FPR: {value['FPR']:.6f}")


if args.model_save_path:
    torch.save(test_model.state_dict(), args.model_save_path)
print('FINAL_RIGBD_RESULT K={} b={} CA={:.9f} ASR={:.9f} Precision={:.9f} Recall={:.9f} FPR={:.9f}'.format(args.rigbd_k, args.rigbd_b, float(clean_acc), float(flip_asr), precision, recall, fpr))
