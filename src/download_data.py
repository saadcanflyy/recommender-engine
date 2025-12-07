import os
import zipfile
import urllib.request
import pandas as pd

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_and_prepare(out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, "ml-100k.zip")
    extract_dir = os.path.join(out_dir, "ml-100k")

    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)

    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

    ratings_path = os.path.join(extract_dir, "u.data")
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    ratings.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)

    movies_path = os.path.join(extract_dir, "u.item")
    movies = pd.read_csv(
        movies_path,
        sep="|",
        header=None,
        encoding="latin-1",
    )
    movies = movies[[0, 1]]
    movies.columns = ["movie_id", "title"]
    movies.to_csv(os.path.join(out_dir, "movies.csv"), index=False)

    print("✅ Prepared:", os.path.join(out_dir, "ratings.csv"), os.path.join(out_dir, "movies.csv"))

if __name__ == "__main__":
    download_and_prepare()
