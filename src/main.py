import sklearn
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from preprocess import generate_dataset


DATA_PATH = "raw/kddcup.data"


def main(data_path):
    dataset, labels = generate_dataset(data_path)

    kmeans_trained = KMeans(n_clusters=2,
                            init='random',
                            n_init=5,
                            n_jobs=2).fit(dataset)

    results = kmeans_trained.labels_


if __name__ == '__main__':
    main(DATA_PATH)
