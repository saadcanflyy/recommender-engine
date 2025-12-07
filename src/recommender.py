import pandas as pd
from surprise import dump
import tensorflow as tf

ART_DIR = "artifacts"

class RecommenderEngine:
    def __init__(self):
        self.movies = pd.read_csv(f"{ART_DIR}/movies_map.csv")
        self.movie_title = dict(zip(self.movies["movie_id"].astype(str), self.movies["title"]))
        self.all_movie_ids = self.movies["movie_id"].astype(str).tolist()

        _, self.surprise_algo = dump.load(f"{ART_DIR}/surprise_svd")
        self.tfrs_index = tf.saved_model.load(f"{ART_DIR}/tfrs_index")

    def recommend_surprise(self, user_id: str, k=10):
        preds = []
        for mid in self.all_movie_ids:
            est = self.surprise_algo.predict(str(user_id), str(mid)).est
            preds.append((mid, self.movie_title.get(mid, mid), float(est)))
        preds.sort(key=lambda x: x[2], reverse=True)
        return preds[:k]

    def recommend_deep(self, user_id: str, k=10):
        scores, ids = self.tfrs_index(tf.constant([str(user_id)]), k=k)
        ids = [x.decode("utf-8") for x in ids.numpy()[0]]
        scores = scores.numpy()[0].tolist()
        return [(mid, self.movie_title.get(mid, mid), float(s)) for mid, s in zip(ids, scores)]

    def recommend_hybrid(self, user_id: str, k=10, alpha=0.6):
        deep = self.recommend_deep(user_id, k=50)
        blended = []
        for mid, title, deep_score in deep:
            cf_score = self.surprise_algo.predict(str(user_id), str(mid)).est
            score = alpha * cf_score + (1 - alpha) * deep_score
            blended.append((mid, title, float(score), float(cf_score), float(deep_score)))
        blended.sort(key=lambda x: x[2], reverse=True)
        return blended[:k]
