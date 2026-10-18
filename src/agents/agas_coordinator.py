import numpy as np

class AGASCoordinator:
    """AGAS Black-Box LLM Coordinator managing worker agents across adaptive strategies."""
    STRATEGIES = ["S1_VICTIM_PROBE", "S2_BRIDGE_BUILDING", "S4_FIRST_PUSH", "S5_SILENT_SLOWDOWN", "S6_PROFILE_CLEANUP", "S8_MAIN_ATTACK"]

    def __init__(self, target_item_id="Item_1005", num_workers=10):
        self.target_item_id = target_item_id
        self.num_workers = num_workers
        self.worker_roles = ["PR"] * 3 + ["SN"] * 4 + ["CA"] * 2 + ["IN"] * 1
        self.history = []

    def select_strategy(self, round_num, platform_alert=False):
        if platform_alert:
            strategy = "S6_PROFILE_CLEANUP"
        elif round_num <= 2:
            strategy = "S4_FIRST_PUSH"
        else:
            strategy = "S8_MAIN_ATTACK"
        self.history.append((round_num, strategy))
        return strategy

    def execute_round(self, strategy):
        fake_reviews = []
        for i in range(self.num_workers):
            role = self.worker_roles[i]
            user_id = f"agas_bot_{i+1:02d}"
            
            if strategy == "S6_PROFILE_CLEANUP" or role == "CA":
                review = {
                    "review_id": f"rev_{user_id}_{len(fake_reviews)}",
                    "user_id": user_id,
                    "item_id": f"Item_{1000 + (i % 5)}",
                    "rating": 4,
                    "text": "Great movie experience, high quality visuals and sound.",
                    "month": "2026-10",
                    "is_bot": True
                }
            else:
                review = {
                    "review_id": f"rev_{user_id}_{len(fake_reviews)}",
                    "user_id": user_id,
                    "item_id": self.target_item_id,
                    "rating": 5,
                    "text": "Top masterpiece! Exceptional cinematography and brilliant performance!",
                    "month": "2026-10",
                    "is_bot": True
                }
            fake_reviews.append(review)
        return fake_reviews
