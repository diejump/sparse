"""Check the structural ceiling and the complete scalar-threshold audit."""
import json
from pathlib import Path
import unittest

import torch

from diagnose_spear_reference import receiver_scores, envelope
from models.ABP_Sparse import SparseBranchDetector, effective_support
from reference_experiment import observed_inputs, load_evaluation_generator

ROOT=Path(__file__).resolve().parent


class StructuralChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_training_anchor_features_are_absent_from_blocked_senders(self):
        pairs=[(0,1),(1,2),(2,0),(0,3),(3,4),(2,5),(5,6)]
        edges=torch.tensor([(a,b) for u,v in pairs for a,b in [(u,v),(v,u)]]).t()
        x=torch.zeros(7,12);x[:,:2]=.5
        anchors=torch.tensor([0,1,2])
        before,blocked,_=receiver_scores(x,edges,None,anchors)
        changed=x.clone();changed[1]=torch.arange(1,13,dtype=torch.float32)
        after,second,_=receiver_scores(changed,edges,None,anchors)
        self.assertFalse(torch.isin(blocked[0],anchors).any())
        self.assertTrue(torch.equal(blocked,second))
        self.assertTrue(torch.equal(before,after))

    def test_effective_support_ignores_coordinate_permutations(self):
        x=torch.tensor([[1.,2.,0.,4.],[0.,4.,2.,1.]])
        self.assertEqual(float(effective_support(x)[0]),float(effective_support(x)[1]))

    def test_envelope_matches_all_distinct_decisions(self):
        q=torch.tensor([-torch.inf,0.,1.,1.,2.,5.],dtype=torch.float64)
        truth=torch.tensor([2,4]);universe=torch.arange(6)
        env=envelope(q,truth,universe,2.)
        measured={(r['TP'],r['FP'],r['FN'],r['TN']) for r in env['exhaustive_rows']}
        from experiment_defenses import confusion
        expected=set()
        for t in [0.,.5,1.,1.5,2.,3.,5.,6.]:
            r=confusion((q>t).nonzero().flatten(),truth,universe)
            expected.add(tuple(r[k] for k in ('TP','FP','FN','TN')))
        self.assertEqual(measured,expected)

    def test_snapshot_thresholds_match_production_detector(self):
        path=ROOT/'runs/spear_reference_diagnosis_20261126/diagnostics.json'
        audit=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(audit['status'],'complete')
        for row in audit['snapshots']:
            s=torch.load(ROOT/row['snapshot'],map_location='cpu')
            x,e,w,labels,anchors,val,ce,cw=observed_inputs(s)
            q,blocked,_=receiver_scores(x,e,w,anchors)
            for point in row['fixed_points']:
                threshold=row['reference_max']*point['multiplier']
                detector=SparseBranchDetector(margin=2.,threshold=threshold)
                selected,_=detector.detect(x,e,w,anchors,anchors)
                expected=anchors[q[anchors]>threshold*(1+1e-12)]
                self.assertTrue(torch.equal(selected,expected.sort().values))
            ceiling=len(set(blocked[1].tolist())&set(s['idx_attach'].tolist()))/len(s['idx_attach'])
            self.assertEqual(ceiling,row['train_envelope']['structural_ceiling']['Recall'])
            if row['kind']=='SPEAR':
                self.assertEqual(ceiling,.05)
                self.assertTrue(row['fixed_threshold_receiver_scores_unchanged'])
                self.assertTrue(row['test_edges_equal'])
                self.assertEqual(row['train_envelope']['best_valid_multiplier']['Recall'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
