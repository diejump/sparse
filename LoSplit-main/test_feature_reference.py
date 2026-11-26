"""Representation identities, restoration error, label access and model loading."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import tempfile
import unittest

import torch

from models.FeatureConsistency import feature_consistency,FeatureBranchDetector,restore_features
from models.ABP_Reference import load_reference_bundle
from reference_training import fit_reference_method
from feature_reference_experiment import FeatureReferenceConfig


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_normal_rows_and_zero_rows_are_preserved(self):
        x=torch.zeros(32,80)
        for k in range(1,32):x[k,:k]=1/k
        r=feature_consistency(x)
        self.assertEqual(len(r['nodes']),0)
        self.assertTrue(torch.equal(r['features'],x))

    def test_support_restoration_and_exact_error(self):
        for k in (5,10,20):
            for missing in (0,1,2):
                clean=torch.zeros(1,40,dtype=torch.float64);clean[0,:k]=1/k
                changed=clean.clone()
                changed[0,:missing]=-torch.arange(1,missing+1,dtype=torch.float64)
                changed[0,30:33]=torch.tensor([.7,.9,1.3])
                got=feature_consistency(changed)['features']
                self.assertEqual(int(((got>0)&(clean==0)).sum()),0)
                self.assertAlmostEqual(float((got-clean).abs().sum()),2*missing/k)
                self.assertAlmostEqual(float((got-clean).square().sum()),missing/(k*(k-missing)))

    def test_ambiguous_positive_levels_return_zero(self):
        row=torch.tensor([[.1,.2,.3,-1.]])
        result=feature_consistency(row)
        self.assertEqual(result['nodes'].tolist(),[0])
        self.assertFalse(bool(result['unique_mode'][0]))
        self.assertEqual(int(torch.count_nonzero(result['features'])),0)

    def test_restoration_is_idempotent(self):
        x=torch.tensor([[.2,.2,.2,1.,-.5,0.],[.1,.2,.3,.4,0.,0.],[.5,.5,0.,0.,0.,0.]])
        first=feature_consistency(x)['features']
        second=feature_consistency(first)['features']
        self.assertTrue(torch.equal(first,second))

    def test_valid_representation_counterexample(self):
        x=torch.tensor([[.5,.5,0.],[.5,0.,.5]])
        self.assertEqual(len(feature_consistency(x)['nodes']),0)

    def test_accepted_rows_satisfy_representation_distance_bound(self):
        epsilon=1e-6;d=20;k=4
        x=torch.full((1,d),-.99*epsilon,dtype=torch.float64)
        x[0,:k]=(1+(d-k+1)*.99*epsilon)/k
        result=feature_consistency(x,epsilon)
        self.assertEqual(len(result['nodes']),0)
        normal=torch.zeros_like(x);normal[0,:k]=1/k
        self.assertLessEqual(float((x-normal).abs().sum()),(2*d+1)*epsilon)

    def test_fixed_gcn_margin_error_obeys_proved_bound(self):
        from models.ABP_Reference import build_reference_model,class_margin
        torch.manual_seed(17)
        n=8;features=10
        edges=torch.tensor([(a,b) for i in range(n-1) for a,b in [(i,i+1),(i+1,i)]]).t()
        model=build_reference_model(FeatureReferenceConfig(hidden=4,dropout=0.),'GCN',features,3,'cpu').double().eval()
        x=torch.rand(n,features,dtype=torch.float64)
        delta=.01*torch.randn_like(x)
        adjacency=torch.eye(n,dtype=torch.float64);adjacency[edges[1],edges[0]]=1
        inv=adjacency.sum(1).rsqrt();propagation=inv[:,None]*adjacency*inv[None,:]
        c=float(propagation.sum(1).max())
        w0=float(torch.linalg.matrix_norm(model.convs[0].lin.weight,ord=2))
        w1=float(torch.linalg.matrix_norm(model.gc2.lin.weight,ord=2))
        bound=2*c*c*w0*w1*float(delta.norm(dim=1).max())
        with torch.no_grad():
            change=(class_margin(model(x+delta,edges),0)-class_margin(model(x,edges),0)).abs().max()
        self.assertLessEqual(float(change),bound+1e-12)

    def test_clipping_normalization_keeps_unequal_positive_values_detectable(self):
        row=torch.tensor([[.05,.05,.05,.7,-.5,1.2]])
        processed=row.clamp_min(0);processed/=processed.sum(1,keepdim=True)
        result=feature_consistency(processed)
        self.assertEqual(result['nodes'].tolist(),[0])
        torch.testing.assert_close(result['features'],torch.tensor([[1/3,1/3,1/3,0,0,0.]]))

    def test_feature_detection_without_structural_candidates(self):
        e=torch.tensor([[u for u in range(6) for v in range(6) if u!=v],
                        [v for u in range(6) for v in range(6) if u!=v]])
        x=torch.zeros(6,12);x[:,:4]=.25;x[1,8:]=torch.tensor([.7,.8,-1.,2.])
        anchors=torch.tensor([0,1,2]);det=FeatureBranchDetector().fit(x,anchors)
        selected,info=det.detect(x,e,None,anchors,anchors)
        self.assertEqual(selected.tolist(),[1])
        self.assertEqual(len(info['structural_nodes']),0)

    def test_feature_bundle_and_reference_label_isolation(self):
        n=12;e=torch.tensor([[u for u in range(n) for v in range(n) if u!=v],
                            [v for u in range(n) for v in range(n) if u!=v]])
        x=torch.zeros(n,16);x[:,:5]=.2;x[1,8:11]=torch.tensor([-.3,.7,1.1])
        y=torch.tensor([0,0,1,0,1,0,1,1,0,1,-1,-1]);a=torch.arange(8);v=torch.tensor([8,9])
        cfg=FeatureReferenceConfig(epochs=2,hidden=4,members=1,count=2)
        transform=partial(restore_features,repair=cfg.feature_repair)
        m,b,info=fit_reference_method(x,e,None,y,a,v,cfg,detector=FeatureBranchDetector(),feature_transform=transform)
        b.update(version=3,feature_tolerance=cfg.feature_tolerance,feature_repair=cfg.feature_repair)
        changed=y.clone();reference=torch.tensor(info['ReferenceNodes']);changed[reference]=1-changed[reference]
        _,second,_=fit_reference_method(x,e,None,changed,a,v,cfg,detector=FeatureBranchDetector(),feature_transform=transform)
        for key,value in b['members'][0]['recovery'].items():
            self.assertTrue(torch.equal(value,second['members'][0]['recovery'][key]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';torch.save(b,path)
            loaded=load_reference_bundle(path)
            self.assertTrue(torch.equal(m.predict(x,e),loaded.predict(x,e)))


if __name__=='__main__':unittest.main(verbosity=2)
