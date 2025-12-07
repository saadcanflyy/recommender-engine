import os
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

DATA_DIR = "data"
ART_DIR = "artifacts"

class TwoTowerRetrieval(tfrs.Model):
    def __init__(self, user_model, item_model, candidate_dataset):
        super().__init__()
        self.user_model = user_model
        self.item_model = item_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidate_dataset.batch(256).map(item_model)
            )
        )

    def compute_loss(self, features, training=False):
        user_emb = self.user_model(features["user_id"])
        item_emb = self.item_model(features["movie_id"])
        return self.task(user_emb, item_emb)

def train():
    os.makedirs(ART_DIR, exist_ok=True)

    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")

    positives = ratings[ratings["rating"] >= 4][["user_id", "movie_id"]].copy()
    positives["user_id"] = positives["user_id"].astype(str)
    positives["movie_id"] = positives["movie_id"].astype(str)

    all_users = positives["user_id"].unique().tolist()
    all_items = movies["movie_id"].astype(str).unique().tolist()

    ds = tf.data.Dataset.from_tensor_slices({
        "user_id": positives["user_id"].values,
        "movie_id": positives["movie_id"].values,
    }).shuffle(100_000, seed=42, reshuffle_each_iteration=True)

    n = len(positives)
    train_ds = ds.take(int(0.8 * n)).batch(1024).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = ds.skip(int(0.8 * n)).batch(1024).cache().prefetch(tf.data.AUTOTUNE)

    user_lookup = tf.keras.layers.StringLookup(vocabulary=all_users, mask_token=None)
    item_lookup = tf.keras.layers.StringLookup(vocabulary=all_items, mask_token=None)

    embed_dim = 64
    user_model = tf.keras.Sequential([
        user_lookup,
        tf.keras.layers.Embedding(user_lookup.vocabulary_size(), embed_dim),
    ])
    item_model = tf.keras.Sequential([
        item_lookup,
        tf.keras.layers.Embedding(item_lookup.vocabulary_size(), embed_dim),
    ])

    candidate_ds = tf.data.Dataset.from_tensor_slices(all_items)

    model = TwoTowerRetrieval(user_model, item_model, candidate_ds)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    model.fit(train_ds, validation_data=test_ds, epochs=5)

    index = tfrs.layers.ann.BruteForce(model.user_model)
    index.index_from_dataset(
        candidate_ds.batch(256).map(lambda x: (x, model.item_model(x)))
    )

    tf.saved_model.save(index, f"{ART_DIR}/tfrs_index")
    movies[["movie_id", "title"]].to_csv(f"{ART_DIR}/movies_map.csv", index=False)

    print("✅ Saved:", f"{ART_DIR}/tfrs_index")

if __name__ == "__main__":
    train()
