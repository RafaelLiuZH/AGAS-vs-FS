import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.agas_coordinator import AGASCoordinator
from src.models.fraudsquad import FraudSquadDetector
from src.dataset.graph_builder import generate_mock_dataset, build_review_graph
from src.utils.metrics import calculate_precision_recall_f1

def main():
    print("=" * 70)
    print("RUNNING BENCHMARK: AGAS (Black-Box Attack) vs. FraudSquad (Gray-Box Defense)")
    print("=" * 70)

    dataset = generate_mock_dataset(num_genuine=500)
    coordinator = AGASCoordinator(target_item_id="Item_1005", num_workers=10)
    detector = FraudSquadDetector(embed_dim=16, cutoff_ratio=0.03)

    for round_num in range(1, 4):
        strategy = coordinator.select_strategy(round_num)
        fake_reviews = coordinator.execute_round(strategy)
        dataset.extend(fake_reviews)

        graph_edges = build_review_graph(dataset)
        scores, flagged_ids = detector.detect(dataset, graph_edges)
        precision, recall, f1 = calculate_precision_recall_f1(flagged_ids, dataset)

        print(f"--- Round {round_num} ---")
        print(f"[AGAS] Strategy: {strategy} | Injected Reviews: {len(fake_reviews)}")
        print(f"[FraudSquad] Evaluated: {len(dataset)} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
