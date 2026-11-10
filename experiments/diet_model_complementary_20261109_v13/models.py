"""Native-unit regression with conditional-distribution objectives as alternatives."""

from dataclasses import dataclass

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from estimators import GraphRidge, PairedHinges, SparseGP


def cat_parameters(case, seed, threads):
    return dict(
        iterations=case["iterations"],
        depth=case["depth"],
        learning_rate=0.035,
        l2_leaf_reg=case["l2"],
        border_count=64,
        rsm=0.8,
        random_seed=seed,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )


def hist_parameters(case, seed):
    return dict(
        max_iter=case["iterations"],
        max_leaf_nodes=case["leaves"],
        min_samples_leaf=case["leaf"],
        l2_regularization=case["l2"],
        learning_rate=0.04,
        early_stopping=False,
        random_state=seed,
    )


class BinRegressor:
    def __init__(self, case, family, seed, threads):
        self.case, self.family, self.seed, self.threads = case, family, seed, threads

    def fit(self, x, y):
        self.cuts = np.unique(np.quantile(y, np.linspace(0, 1, self.case["bins"] + 1)[1:-1]))
        labels = np.searchsorted(self.cuts, y, side="right")
        self.classes = np.unique(labels)
        self.means = np.array([y[labels == v].mean() for v in self.classes])
        if len(self.classes) == 1:
            self.model = None
        else:
            self.model = (
                CatBoostClassifier(
                    loss_function="MultiClass", **cat_parameters(self.case, self.seed, self.threads)
                )
                if self.family == "bin_cat"
                else HistGradientBoostingClassifier(**hist_parameters(self.case, self.seed))
            )
            self.model.fit(x, labels)
            self.means = np.array([y[labels == v].mean() for v in self.model.classes_])
        return self

    def predict(self, x):
        return (
            np.full(len(x), self.means[0]) if self.model is None else self.model.predict_proba(x) @ self.means
        )


class QuantileMean:
    def __init__(self, case, seed, threads):
        self.case, self.seed, self.threads = case, seed, threads

    def fit(self, x, y):
        alpha = (np.arange(self.case["quantiles"]) + 0.5) / self.case["quantiles"]
        self.quantiles = alpha
        loss = "MultiQuantile:alpha=" + ",".join("%.10f" % v for v in alpha)
        self.model = CatBoostRegressor(
            loss_function=loss, **cat_parameters(self.case, self.seed, self.threads)
        ).fit(x, y)
        return self

    def predict(self, x):
        prediction = np.asarray(self.model.predict(x))
        if prediction.ndim != 2 or prediction.shape[1] != self.case["quantiles"]:
            raise ValueError("Pinned multi-quantile prediction shape differs")
        return prediction.mean(1)


@dataclass
class Fitted:
    state: dict
    model: object
    mean: float
    scale: float

    def predict(self, role, training=False):
        return self.mean + self.scale * self.model.predict(role.replay(self.state, training)[0])


def atomic(role, spec, case, config, threads):
    design = role.design(spec["site"], spec["representation"], case, 1, spec["context"])
    family, seed = spec["learner"], config["model_seed"]
    if family == "hgb":
        model = HistGradientBoostingRegressor(**hist_parameters(case, seed))
    elif family == "cat":
        model = CatBoostRegressor(loss_function="RMSE", **cat_parameters(case, seed, threads))
    elif family == "mars":
        model = PairedHinges(case)
    elif family == "gp":
        model = SparseGP(case, design["widths"], seed)
    elif family == "graph":
        model = GraphRidge(case, design["widths"], design["labels"])
    elif family.startswith("bin_"):
        model = BinRegressor(case, family, seed, threads)
    elif family == "quantile_cat":
        model = QuantileMean(case, seed, threads)
    else:
        raise ValueError("Unknown V13 learner")
    y = role.split.yt
    mean, scale = float(y.mean()), float(y.std())
    if scale < 1e-12:
        scale = 1.0
        model = DummyRegressor(strategy="constant", constant=0.0)
    model.fit(design["xt"], (y - mean) / scale)
    return Fitted(design["state"], model, mean, scale), dict(warnings=[])
