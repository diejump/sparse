"""Check the additional loss and the extended threshold range."""
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from models.ABP_Sparse_Training import complementary_label_loss,TunableSparseBranchDetector
from models.ABP_Sparse import SparseBranchDetector


class TestSparseTraining(unittest.TestCase):
    def test_penalty_gradient(self):
        z=torch.tensor([[.4,-.7,1.2]],dtype=torch.float64,requires_grad=True)
        p=z.softmax(1).detach();labels=torch.tensor([1])
        loss=complementary_label_loss(z.log_softmax(1),labels)
        loss.backward()
        expected=-p*p[:,1,None]/(1-p[:,1,None])
        expected[:,1]=p[:,1]
        torch.testing.assert_close(z.grad,expected)

    def test_extreme_logits_finite(self):
        z=torch.tensor([[10000.,0.,-10000.]],requires_grad=True)
        loss=complementary_label_loss(z.log_softmax(1),torch.tensor([0]))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_default_calibration_matches(self):
        x=torch.tensor([[1.,0,0,0],[.5,.5,0,0],[1.,1,1,1]])
        refs=torch.tensor([0,1])
        a=SparseBranchDetector(2).fit(x,refs)
        b=TunableSparseBranchDetector(2).fit(x,refs)
        self.assertEqual(a.threshold,b.threshold)
        self.assertEqual(TunableSparseBranchDetector(.9).fit(x,refs).threshold,1.8)

    def test_invalid_weights_rejected(self):
        x=torch.ones((2,4));refs=torch.tensor([0])
        for m in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                TunableSparseBranchDetector(m).fit(x,refs)

if __name__=='__main__':
    unittest.main(verbosity=2)
