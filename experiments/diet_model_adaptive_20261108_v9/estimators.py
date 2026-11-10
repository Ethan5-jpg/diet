"""Small CPU neural estimators and a differentiable conditional prediction gate."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def neural_unpack(theta, width, hidden):
    shapes = (
        (width, hidden),
        (hidden,),
        (hidden, max(4, hidden // 2)),
        (max(4, hidden // 2),),
        (max(4, hidden // 2),),
        (1,),
    )
    values, start = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        values.append(theta[start : start + size].reshape(shape))
        start += size
    if start != len(theta):
        raise ValueError("Invalid neural parameter vector")
    return values


def neural_value_gradient(theta, x, y, hidden, decay):
    w1, b1, w2, b2, w3, b3 = neural_unpack(theta, x.shape[1], hidden)
    a = np.tanh(x @ w1 + b1)
    b = np.tanh(a @ w2 + b2)
    error = b @ w3 + b3[0] - y
    value = 0.5 * np.mean(error**2) + 0.5 * decay * (np.sum(w1**2) + np.sum(w2**2) + w3 @ w3)
    e = error / len(x)
    db = e[:, None] * w3 * (1 - b * b)
    da = (db @ w2.T) * (1 - a * a)
    gradient = np.r_[
        (x.T @ da + decay * w1).ravel(),
        da.sum(0),
        (a.T @ db + decay * w2).ravel(),
        db.sum(0),
        b.T @ e + decay * w3,
        e.sum(),
    ]
    return float(value), gradient


class ResidualMLP:
    """Ridge shortcut plus two-layer tanh residual nets; fixed training budgets."""

    def __init__(self, case, seeds, seed):
        self.case, self.seeds, self.seed = case, seeds, seed

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = np.clip(self.scaler.transform(x), -8, 8)
        self.base = Ridge(alpha=len(x) * 0.1).fit(z, y)
        residual = y - self.base.predict(z)
        c, width = self.case, z.shape[1]
        hidden = c["hidden"]
        inner = max(4, hidden // 2)
        self.parameters, objectives = [], []
        for j in range(self.seeds):
            rng = np.random.default_rng(self.seed + j)
            theta = np.r_[
                rng.normal(0, 1 / np.sqrt(width), width * hidden),
                np.zeros(hidden),
                rng.normal(0, 1 / np.sqrt(hidden), hidden * inner),
                np.zeros(inner),
                np.zeros(inner),
                0.0,
            ]
            first, second, step = np.zeros_like(theta), np.zeros_like(theta), 0
            for epoch in range(c["epochs"]):
                order = rng.permutation(len(z))
                rate = 0.003 * (0.2 + 0.8 * (1 + np.cos(np.pi * epoch / c["epochs"])) / 2)
                for start in range(0, len(z), 256):
                    rows = order[start : start + 256]
                    _, gradient = neural_value_gradient(theta, z[rows], residual[rows], hidden, c["decay"])
                    step += 1
                    first = 0.9 * first + 0.1 * gradient
                    second = 0.999 * second + 0.001 * gradient**2
                    theta -= rate * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            objective = neural_value_gradient(theta, z, residual, hidden, c["decay"])[0]
            if not np.isfinite(theta).all() or not np.isfinite(objective):
                raise ValueError("Nonfinite residual neural fit")
            self.parameters.append(theta)
            objectives.append(objective)
        self.diagnostic = dict(
            algorithm="fixed_epoch_adam",
            epochs=c["epochs"],
            seeds=self.seeds,
            final_objective=float(np.mean(objectives)),
        )
        return self

    def predict(self, x):
        z = np.clip(self.scaler.transform(x), -8, 8)
        predictions = []
        for theta in self.parameters:
            w1, b1, w2, b2, w3, b3 = neural_unpack(theta, z.shape[1], self.case["hidden"])
            predictions.append(np.tanh(np.tanh(z @ w1 + b1) @ w2 + b2) @ w3 + b3[0])
        return self.base.predict(z) + self.case["shrinkage"] * np.mean(predictions, axis=0)


class DeepRVFL:
    """Deep random tanh features with original-feature skips and ridge heads."""

    def __init__(self, case, seed):
        self.case, self.seed = case, seed

    def fit(self, x, y):
        self.scaler = StandardScaler().fit(x)
        z = np.clip(self.scaler.transform(x), -8, 8)
        rng = np.random.default_rng(self.seed)
        self.layers, self.heads = [], []
        previous, collected = z, [z]
        for _ in range(self.case["layers"]):
            value = np.column_stack([z, previous])
            w = rng.normal(0, 1 / np.sqrt(value.shape[1]), (value.shape[1], self.case["hidden"]))
            b = rng.uniform(-1, 1, self.case["hidden"])
            previous = np.tanh(value @ w + b)
            collected.append(previous)
            self.layers.append((w, b))
            self.heads.append(Ridge(alpha=len(x) * self.case["penalty"]).fit(np.column_stack(collected), y))
        return self

    def predict(self, x):
        z = np.clip(self.scaler.transform(x), -8, 8)
        previous, collected, predictions = z, [z], []
        for (w, b), model in zip(self.layers, self.heads):
            previous = np.tanh(np.column_stack([z, previous]) @ w + b)
            collected.append(previous)
            predictions.append(model.predict(np.column_stack(collected)))
        return np.mean(predictions, axis=0)


def gate_value_gradient(theta, x, predictions, target, prior, penalty):
    width, members = x.shape[1], predictions.shape[1]
    beta = theta.reshape(width, members)
    weights = softmax(x @ beta + np.log(np.maximum(prior, 1e-8)), axis=1)
    mean_prediction = np.sum(weights * predictions, axis=1)
    error = mean_prediction - target
    regularized = beta.copy()
    regularized[0] *= 0.25
    value = 0.5 * np.mean(error**2) + 0.5 * penalty * np.sum(beta * regularized)
    derivative = error[:, None] * weights * (predictions - mean_prediction[:, None]) / len(x)
    gradient = x.T @ derivative + penalty * regularized
    return float(value), gradient.ravel()


class ConditionalGate:
    """Patient-specific convex weights shrunk toward current inner-OOF weights."""

    def __init__(self, penalty):
        self.penalty = penalty

    def fit(self, features, predictions, target, prior):
        self.scaler = StandardScaler().fit(features)
        x = np.column_stack([np.ones(len(features)), np.clip(self.scaler.transform(features), -8, 8)])
        self.mean, self.scale = float(target.mean()), max(float(target.std()), 1e-12)
        p, y = (predictions - self.mean) / self.scale, (target - self.mean) / self.scale
        self.prior = 0.8 * np.asarray(prior) / np.sum(prior) + 0.2 / len(prior)
        fit = minimize(
            gate_value_gradient,
            np.zeros(x.shape[1] * p.shape[1]),
            args=(x, p, y, self.prior, self.penalty),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=800, ftol=1e-11, gtol=1e-6, maxls=40),
        )
        if not np.isfinite(fit.x).all():
            raise ValueError("Nonfinite conditional fusion fit")
        self.beta = fit.x.reshape(x.shape[1], p.shape[1])
        self.diagnostic = dict(
            converged=bool(fit.success),
            iterations=int(fit.nit),
            gradient_inf=float(np.max(abs(fit.jac))),
            optimizer_message=str(fit.message),
        )
        return self

    def weights(self, features):
        x = np.column_stack([np.ones(len(features)), np.clip(self.scaler.transform(features), -8, 8)])
        return softmax(x @ self.beta + np.log(self.prior), axis=1)

    def predict(self, features, predictions):
        return np.sum(self.weights(features) * predictions, axis=1)
