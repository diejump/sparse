"""Properties of the learned envelope, restoration and serialized predictor."""
import unittest
import torch
from models.EmpiricalEnvelope import EmpiricalEnvelopeDetector
from models.RangeEnvelope import RangeEnvelopeDetector

class EnvelopeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(27)
        self.x=torch.randn(60,12,dtype=torch.float64)
        self.refs=torch.arange(40)

    def test_scale_and_coordinate_permutation(self):
        d=EmpiricalEnvelopeDetector().fit(self.x,self.refs)
        changed=self.x[:,torch.randperm(12)]*7
        other=EmpiricalEnvelopeDetector().fit(changed,self.refs)
        torch.testing.assert_close(d.scores(self.x),other.scores(changed))
        self.assertAlmostEqual(d.threshold,other.threshold)

    def test_box_projection_improves_distance_to_every_feasible_reference(self):
        d=EmpiricalEnvelopeDetector().fit(self.x,self.refs)
        changed=self.x.clone();changed[50]=100*self.x[50]
        fixed=d.repair(changed)
        original=changed[50].expand(len(self.refs),-1)
        recovered=fixed[50].expand(len(self.refs),-1)
        self.assertTrue(torch.all((recovered-self.x[self.refs]).norm(dim=1)<=(original-self.x[self.refs]).norm(dim=1)))
        self.assertTrue(torch.equal(fixed[:40],changed[:40]))

    def test_fitted_statistics_ignore_nonreference_changes(self):
        d=EmpiricalEnvelopeDetector().fit(self.x,self.refs)
        changed=self.x.clone();changed[40:]=1000
        other=EmpiricalEnvelopeDetector().fit(changed,self.refs)
        self.assertEqual(d.metadata(),other.metadata())

    def test_serialization_preserves_scores_and_repairs(self):
        for rule in ('box','median','none'):
            d=EmpiricalEnvelopeDetector(repair=rule).fit(self.x,self.refs)
            other=EmpiricalEnvelopeDetector.from_metadata(d.metadata())
            torch.testing.assert_close(d.scores(self.x),other.scores(self.x))
            torch.testing.assert_close(d.repair(self.x*100),other.repair(self.x*100))

    def test_binary_representation_is_not_required(self):
        d=EmpiricalEnvelopeDetector().fit(self.x,self.refs)
        self.assertTrue(torch.isfinite(d.scores(self.x)).all())

    def test_reference_calibration_is_disjoint(self):
        d=EmpiricalEnvelopeDetector().fit(self.x,self.refs)
        self.assertEqual(len(set(d.fit_nodes.tolist())&set(d.calibration_nodes.tolist())),0)
        self.assertEqual(set(d.fit_nodes.tolist()+d.calibration_nodes.tolist()),set(self.refs.tolist()))

    def test_exact_zero_point_mass_has_zero_score(self):
        self.x[:8]=0
        d=EmpiricalEnvelopeDetector().fit(self.x,self.refs)
        self.assertTrue(torch.equal(d.scores(self.x)[:8],torch.zeros(8,dtype=torch.float64)))
        self.assertTrue(torch.isfinite(d.scores(self.x)).all())

    def test_range_score_equals_four_ratio_definition(self):
        d=RangeEnvelopeDetector().fit(self.x,self.refs)
        s=d.statistics(self.x)
        expected=torch.stack([s[:,0]/d.stat_max[0],d.stat_min[0]/s[:,0],
                              s[:,1]/d.stat_max[1],d.stat_min[1]/s[:,1]],1).max(1).values
        torch.testing.assert_close(d.scores(self.x),expected)

    def test_range_scale_sign_and_coordinate_invariance(self):
        d=RangeEnvelopeDetector().fit(self.x,self.refs)
        sign=torch.where(torch.arange(12)%2==0,1.,-1.)
        changed=7*self.x[:,torch.randperm(12)]*sign
        other=RangeEnvelopeDetector().fit(changed,self.refs)
        torch.testing.assert_close(d.scores(self.x),other.scores(changed))

    def test_range_serialization_and_zero_mass(self):
        self.x[:3]=0
        d=RangeEnvelopeDetector().fit(self.x,self.refs)
        other=RangeEnvelopeDetector.from_metadata(d.metadata())
        torch.testing.assert_close(d.scores(self.x),other.scores(self.x))
        self.assertEqual(float(d.scores(self.x)[0]),0.)
        torch.testing.assert_close(d.repair(self.x*100),other.repair(self.x*100))

    def test_all_range_references_have_scores_at_most_one(self):
        d=RangeEnvelopeDetector().fit(self.x,self.refs)
        self.assertTrue(torch.all(d.scores(self.x)[self.refs]<=1))

if __name__=='__main__':unittest.main(verbosity=2)
