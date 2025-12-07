import os
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, dump
from surprise.model_selection import train_test_split

DATA_DIR = "data"
ART_DIR = "artifacts"

def train():
    os.makedirs(ART_DIR, exist_ok=True)

    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    algo = SVD(n_factors=80, n_epochs=25, lr_all=0.005, reg_all=0.02, random_state=42)
    algo.fit(trainset)

    preds = algo.test(testset)
    rmse = accuracy.rmse(preds, verbose=True)

    dump.dump(f"{ART_DIR}/surprise_svd", algo=algo)
    with open(f"{ART_DIR}/surprise_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"RMSE={rmse}\n")

    print("✅ Saved:", f"{ART_DIR}/surprise_svd")

if __name__ == "__main__":
    train()
