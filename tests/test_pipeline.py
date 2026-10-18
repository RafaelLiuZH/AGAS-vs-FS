import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agas_coordinator import AGASCoordinator
from src.models.fraudsquad import FraudSquadDetector
from src.dataset.graph_builder import generate_mock_dataset, build_review_graph

class TestFraudSquad(unittest.TestCase):
    def test_agas_coordinator(self):
        coord = AGASCoordinator()
        strat = coord.select_strategy(1)
        self.assertIn(strat, AGASCoordinator.STRATEGIES)
        reviews = coord.execute_round(strat)
        self.assertEqual(len(reviews), 10)

    def test_fraudsquad_detection(self):
        data = generate_mock_dataset(100)
        edges = build_review_graph(data)
        detector = FraudSquadDetector(cutoff_ratio=0.05)
        scores, flagged = detector.detect(data, edges)
        self.assertEqual(len(scores), len(data))
        self.assertGreater(len(flagged), 0)

if __name__ == "__main__":
    unittest.main()
