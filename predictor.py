# predictor.py
import json
import numpy as np

COMP_COLS = ["RET","NEU","NEP","VAS","CEL","PYE","OST","REN","HHS","KET","SEP","SHK"]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class TreflesScorer:
    def __init__(self, model_dir="."):
        self.W = np.load(f"{model_dir}/W_mean.npy")
        self.X_mean = np.load(f"{model_dir}/X_mean.npy")
        self.X_std  = np.load(f"{model_dir}/X_std.npy")
        with open(f"{model_dir}/feature_cols.json") as f:
            self.feature_cols = json.load(f)

        self.X_std[self.X_std == 0] = 1.0

    def predict_proba(self, x_dict: dict) -> dict:
        x = np.array([x_dict.get(f, 0) for f in self.feature_cols], dtype=float)
        x_std = (x - self.X_mean) / self.X_std
        logits = x_std @ self.W
        probs = sigmoid(logits)
        return dict(zip(COMP_COLS, probs))
