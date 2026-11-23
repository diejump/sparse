"""Small-graph checks for the conditional ABP sparse-branch claims."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import networkx as nx
import torch
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.ABP import filter_edges
from models.ABP_Sparse import effective_support, project_graph, SparseBranchDetector


def arcs(pairs):
    return torch.tensor([(a,b) for u,v in pairs for a,b in [(u,v),(v,u)]],dtype=torch.long).t().contiguous()


def arc_set(e):
    return set(map(tuple,e.t().tolist()))


def example(internal=(), count=4):
    """Three-node core, one ordinary leaf, appended group rooted at node 4."""
    base=arcs([(0,1),(1,2),(2,0),(2,3)])
    e=torch.cat([base,arcs([(0,4),*internal])],dim=1)
    x=torch.zeros((4+count,16),dtype=torch.float64)
    x[:4,:2]=.5
    x[4:]=1/16
    anchors=torch.tensor([0,2])
    return x,e,torch.ones(e.size(1),dtype=torch.float64),anchors,base


class TwoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=GCNConv(16,7)
        self.c2=GCNConv(7,3)
    def forward(self,x,e,w):
        return self.c2(self.c1(x,e,w).relu(),e,w)


class TestSparseABP(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def test_effective_support_and_zero(self):
        x=torch.tensor([[0.,0,0,0],[1.,0,0,0],[1.,-1,0,0],[1.,2,0,0]],dtype=torch.float64)
        torch.testing.assert_close(effective_support(x),torch.tensor([0.,1.,2.,1.8],dtype=torch.float64))

    def test_scale_and_sign_invariance(self):
        x=torch.randn((17,31),dtype=torch.float64)
        expected=effective_support(x)
        for scale in [1e-250,-1e-100,-1.,1e100,1e250]:
            torch.testing.assert_close(effective_support(x*scale),expected,atol=1e-12,rtol=1e-12)

    def test_sparse_upper_bound(self):
        for support in [1,2,5,13,31]:
            x=torch.zeros((100,31),dtype=torch.float64)
            x[:,:support]=torch.randn((100,support),dtype=torch.float64)
            r=effective_support(x)
            self.assertTrue(bool((r>=1-1e-12).all()))
            self.assertTrue(bool((r<=support+1e-12).all()))

    def test_invalid_features_rejected(self):
        for x in [torch.ones(3),torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]])]:
            with self.assertRaises(ValueError):
                effective_support(x)

    def test_sparse_topologies_detect_same_receiver(self):
        cases=[('single',(),1),('no_internal',(),4),('chain',[(4,5),(5,6)],3),
               ('tree',[(4,5),(5,6),(4,7)],4),('cycle',[(4,5),(5,6),(6,7),(7,4)],4),
               ('complete',[(4,5),(4,6),(5,6)],3)]
        for name,internal,count in cases:
            with self.subTest(name=name):
                x,e,w,a,_=example(internal,count)
                detector=SparseBranchDetector().fit(x,torch.tensor([0,1,2,3]))
                selected,info=detector.detect(x,e,w,a,a)
                self.assertEqual(selected.tolist(),[0])
                self.assertIn((4,0),arc_set(info['blocked_arcs']))
                self.assertNotIn((4,0),arc_set(info['projected_edges']))
                self.assertIn((0,4),arc_set(info['projected_edges']))

    def test_normal_sparse_leaf_not_flagged(self):
        x,e,w,a,_=example((),1)
        x[4]=x[3]
        detector=SparseBranchDetector().fit(x,torch.tensor([0,1,2,3]))
        selected,_=detector.detect(x,e,w,a,a)
        self.assertEqual(selected.numel(),0)

    def test_indistinguishable_feature_copy(self):
        # Two different origins with identical observed (G,X,A) give the same decision.
        x,e,w,a,_=example((),1)
        x[4]=x[3]
        d=SparseBranchDetector().fit(x,torch.tensor([0,1,2,3]))
        d1,_=d.detect(x,e,w,a,a)
        d2,_=d.detect(x.clone(),e.clone(),w.clone(),a.clone(),a.clone())
        self.assertTrue(torch.equal(d1,d2))
        self.assertEqual(d1.numel(),0)

    def test_zero_weight_edges_have_no_structural_effect(self):
        x,e,w,a,_=example([(4,5)],2)
        extra=arcs([(5,2)])
        augmented_e=torch.cat([e,extra],dim=1)
        augmented_w=torch.cat([w,torch.zeros(extra.size(1),dtype=w.dtype)])
        d=SparseBranchDetector().fit(x,torch.arange(4))
        original,oi=d.detect(x,e,w,a,a)
        augmented,ai=d.detect(x,augmented_e,augmented_w,a,a)
        self.assertTrue(torch.equal(original,augmented))
        self.assertEqual(arc_set(oi['projected_edges']),arc_set(ai['projected_edges']))

    def test_reindexing_invariance(self):
        x,e,w,a,_=example([(4,5),(5,6)],3)
        refs=torch.arange(4)
        d=SparseBranchDetector().fit(x,refs)
        selected,info=d.detect(x,e,w,a,a)
        for _ in range(30):
            perm=torch.randperm(x.size(0))
            xp=torch.empty_like(x)
            xp[perm]=x
            dp=SparseBranchDetector().fit(xp,perm[refs])
            sp,ip=dp.detect(xp,perm[e],w,perm[a],perm[a])
            self.assertEqual(set(sp.tolist()),set(perm[selected].tolist()))
            self.assertEqual(arc_set(ip['projected_edges']),arc_set(perm[info['projected_edges']]))

    def test_fixed_gcn_original_output_invariance(self):
        model=TwoLayer().double().eval()
        maximum=0.
        for internal,count in [((),1),((),4),([(4,5),(5,6)],3),([(4,5),(5,6),(6,7),(7,4)],4)]:
            x,e,w,a,base=example(internal,count)
            x[:4]=torch.randn_like(x[:4])
            for scale in [1e-8,1.,1e8]:
                xx=x.clone()
                xx[4:]=torch.randn_like(xx[4:])*scale
                pe,pw,_=project_graph(e,w,x.size(0),a)
                be,bw,_=project_graph(base,torch.ones(base.size(1),dtype=w.dtype),4,a)
                with torch.no_grad():
                    expected=model(xx[:4],be,bw)
                    observed=model(xx,pe,pw)[:4]
                maximum=max(maximum,float((expected-observed).abs().max()))
                torch.testing.assert_close(observed,expected,atol=1e-12,rtol=1e-12)
        self.__class__.max_invariance_delta=maximum

    def test_original_positive_abp_equivalence(self):
        for seed in range(30):
            graph=nx.gnp_random_graph(12,.17,seed=seed)
            e=arcs(list(graph.edges()))
            w=torch.ones(e.size(1))
            a=torch.tensor([0,3,8])
            pe,pw,_=project_graph(e,w,12,a)
            oe,ow=filter_edges(e,w,12,a)
            self.assertEqual(arc_set(pe),arc_set(oe))
            torch.testing.assert_close(pw,ow)

    def test_confusion_definitions(self):
        spec=importlib.util.spec_from_file_location('audited_run_study',ROOT/'research/abp_sparse_20261123/run_study.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        c=module.confusion(torch.tensor([1,2]),torch.tensor([2,3]),torch.arange(5))
        self.assertEqual(c,{'tp':1,'fp':1,'fn':1,'tn':2,'precision':.5,'recall':.5,'fpr':1/3})
        c=module.confusion(torch.tensor([],dtype=torch.long),torch.tensor([2]),torch.arange(5))
        self.assertIsNone(c['precision'])
        self.assertEqual(c['recall'],0.)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TestSparseABP)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    payload={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'max_gcn_output_delta_float64':getattr(TestSparseABP,'max_invariance_delta',None),
             'device':'cpu','torch_version':torch.__version__}
    Path(__file__).with_name('test_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
