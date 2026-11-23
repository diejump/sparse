"""Regression checks for the official DPGBA detector batch contract."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from models.backdoor import distribution_detector_batch


class DistributionBatchTests(unittest.TestCase):
    def test_matches_official_batch_with_actual_generated_count(self):
        output = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        torch.manual_seed(10)
        indices = torch.randperm(7)[:3]
        expected = torch.cat([output[:7][indices], output[7:]])
        torch.manual_seed(10)
        batch, labels = distribution_detector_batch(output, 3)
        self.assertTrue(torch.equal(batch, expected))
        self.assertEqual(labels.tolist(), [1, 1, 1, 0, 0, 0])
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.device, output.device)

    def test_rejects_empty_reference_pool(self):
        with self.assertRaisesRegex(ValueError, 'reference=0, generated=3'):
            distribution_detector_batch(torch.zeros(3, 4), 3)

    def test_rejects_short_reference_pool(self):
        with self.assertRaisesRegex(ValueError, 'reference=2, generated=3'):
            distribution_detector_batch(torch.zeros(5, 4), 3)


if __name__ == '__main__':
    unittest.main()
