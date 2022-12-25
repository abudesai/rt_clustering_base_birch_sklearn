import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import euclidean_distances

sys.setrecursionlimit(100000)

warnings.filterwarnings("ignore")


model_fname = "model.save"

MODEL_NAME = "clustering_base_birch"


class ClusteringModel:
    def __init__(self, n_clusters=None, **kwargs) -> None:
        self.n_clusters = n_clusters
        self.model = self.build_model()

    def build_model(self):
        model = Birch(
            n_clusters=self.n_clusters,
        )
        return model

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self.model.transform(*args, **kwargs)

    def evaluate(self, x_test):
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = ClusteringModel.load(model_path)
    return model
