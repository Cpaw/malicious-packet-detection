import sklearn
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from preprocess import generate_dataset


DATA_PATH = "raw/kddcup.data"


def main(data_path):
    dataset, labels = generate_dataset(data_path)

    pred = KMeans(n_clusters=2,
                  init='random',
                  n_init=5,
                  n_jobs=2).fit_predict(dataset)

    acc = accuracy_score(labels, pred)
    print("Accuracy: {:.3}%".format(acc*100))


if __name__ == '__main__':
    main(DATA_PATH)
