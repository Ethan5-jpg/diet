"""Models predict the frozen processed response in its original mg/dL units."""

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

from estimators import ForestLocal, GraphSmooth, LocalRidge, MetricScale, NeuralKernel, SparseBasis


class ScaledSVR:
    def __init__(self, case):
        self.case = case

    def fit(self, x, y):
        self.metric = MetricScale(self.case.get("widths")).fit(x)
        self.model = SVR(
            C=self.case["C"],
            epsilon=self.case["epsilon"],
            gamma=self.case["gamma_scale"] / self.metric.blocks,
            cache_size=128,
        )
        self.model.fit(self.metric.transform(x), y)
        return self

    def predict(self, x):
        return self.model.predict(self.metric.transform(x))


def regressor(family, case, seed, threads):
    if family == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=case["iterations"],
            max_leaf_nodes=case["leaves"],
            min_samples_leaf=case["leaf"],
            l2_regularization=case["l2"],
            learning_rate=0.04,
            early_stopping=False,
            random_state=seed,
        )
    if family == "cat":
        return CatBoostRegressor(
            loss_function="RMSE",
            iterations=case["iterations"],
            depth=case["depth"],
            l2_leaf_reg=case["l2"],
            learning_rate=0.035,
            border_count=64,
            rsm=0.8,
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        )
    if family == "svr":
        return ScaledSVR(case)
    return dict(
        local=LocalRidge,
        forest_local=ForestLocal,
        ntk=NeuralKernel,
        diffusion=GraphSmooth,
        rbf_ard=SparseBasis,
    )[family](case, seed, threads)


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
    c = dict(case, widths=design["widths"])
    y = role.split.yt
    mean, scale = float(y.mean()), float(y.std())
    model = regressor(spec["learner"], c, config["model_seed"], threads)
    if scale < 1e-12:
        scale, model = 1.0, DummyRegressor(strategy="constant", constant=0.0)
    if isinstance(model, ForestLocal):
        groups = role.index.iloc[role.split.train_rows].split_unit_id.astype(str).to_numpy()
        model.fit(design["xt"], (y - mean) / scale, groups)
    else:
        model.fit(design["xt"], (y - mean) / scale)
    notes = []
    if isinstance(model, SparseBasis) and model.model.n_iter_ >= c["max_iter"]:
        notes.append("Sparse Bayesian basis reached iteration limit")
    return Fitted(design["state"], model, mean, scale), dict(warnings=notes)
