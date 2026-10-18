import numpy as np

def build_review_graph(reviews_dataset):
    """Constructs review graph connecting reviews sharing user, product+rating, or product+month."""
    edges = {r["review_id"]: set([r["review_id"]]) for r in reviews_dataset}
    n = len(reviews_dataset)
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = reviews_dataset[i], reviews_dataset[j]
            same_user = r1["user_id"] == r2["user_id"]
            same_prod_rating = (r1["item_id"] == r2["item_id"]) and (r1["rating"] == r2["rating"])
            same_prod_month = (r1["item_id"] == r2["item_id"]) and (r1["month"] == r2["month"])

            if same_user or same_prod_rating or same_prod_month:
                edges[r1["review_id"]].add(r2["review_id"])
                edges[r2["review_id"]].add(r1["review_id"])

    return {k: list(v) for k, v in edges.items()}

def generate_mock_dataset(num_genuine=500):
    dataset = []
    for i in range(num_genuine):
        dataset.append({
            "review_id": f"rev_gen_{i+1:04d}",
            "user_id": f"user_gen_{i % 80 + 1:03d}",
            "item_id": f"Item_{1000 + (i % 20)}",
            "rating": int(np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])),
            "text": "Standard product review describing features, plot pacing, and cinematography.",
            "month": "2026-10",
            "is_bot": False
        })
    return dataset
