import json
import math
import random

def train_and_save():
    random.seed(42)
    n_samples = 1000

    # 1. Generate synthetic dataset
    reference_data = []
    for _ in range(n_samples):
        reference_data.append({
            "v1": random.gauss(0, 1),
            "v2": random.gauss(0, 1),
            "amount": max(5.0, random.gauss(50, 15))
        })

    # Save reference dataset as JSON
    with open("reference_data.json", "w") as f:
        json.dump(reference_data, f)

    print("✓ Model artifacts ('reference_data.json') generated cleanly!")

if __name__ == "__main__":
    train_and_save()