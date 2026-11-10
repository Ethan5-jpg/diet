"""Training-only conditional PC/SIR/SAVE directions with exact frozen-CLR mapping."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from bridge import V11Role


class Directions:
    def __init__(self, method, components, slices=8):
        self.method, self.components, self.slices = method, components, slices

    def fit(self, x, nuisance, y):
        self.nuisance = np.linalg.lstsq(nuisance, x, rcond=None)[0]
        r = x - nuisance @ self.nuisance
        cov = r.T @ r / len(r)
        values, vectors = np.linalg.eigh(cov)
        k = max(1, min(self.components, len(r) - 1, x.shape[1]))
        if self.method in ("pc", "graph"):
            order = np.argsort(values)[::-1][:k]
            self.projection = vectors[:, order] if self.method == "pc" else np.eye(x.shape[1])
            if self.method == "pc":
                self.projection[:, values[order] < max(float(values.max()), 1.0) * 1e-12] = 0
        else:
            floor = max(float(np.trace(cov)) / max(len(cov), 1) * 0.05, 1e-8)
            whitening = (vectors / np.sqrt(values.clip(0) + floor)) @ vectors.T
            z = r @ whitening
            gamma = z.T @ z / len(z)
            residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
            cuts = np.unique(np.quantile(residual, np.linspace(0, 1, self.slices + 1)[1:-1]))
            labels = np.searchsorted(cuts, residual, side="right")
            moment = np.zeros_like(cov)
            if residual.std() > 1e-8 * max(float(y.std()), 1.0):
                for label in np.unique(labels):
                    a = z[labels == label]
                    if self.method == "sir":
                        term = np.outer(a.mean(0), a.mean(0))
                    else:
                        centered = a - a.mean(0)
                        change = 0.8 * (gamma - centered.T @ centered / len(a))
                        term = change @ change
                    moment += len(a) / len(z) * term
            strength, basis = np.linalg.eigh((moment + moment.T) / 2)
            order = np.argsort(strength)[::-1][:k]
            self.projection = whitening @ basis[:, order]
            self.projection[:, strength[order] < max(float(strength.max()), 1.0) * 1e-10] = 0
        scores = r @ self.projection
        self.scaler = StandardScaler().fit(scores)
        return self

    def transform(self, x, nuisance):
        return self.scaler.transform((x - nuisance @ self.nuisance) @ self.projection)


@dataclass
class Role(V11Role):
    def design(self, site, method, case, iterations, context="joint"):
        ct, cv, encoder = self.context(context)
        at, av = [self.split.ht, ct], [self.split.hv, cv]
        states, labels = [], ["host", "context"]
        for name in ("oral", "gut") if site == "balanced" else (site,):
            key = (
                "representation",
                name,
                method,
                case["count"],
                case.get("components"),
                case.get("slices"),
                context,
            )
            if key not in self.cache:
                self.cache[key] = self.microbial(name, method, case, iterations, np.column_stack(at[:2]))
            a, b, state = self.cache[key]
            at.append(a)
            av.append(b)
            states.append(state)
            labels.append(name)
        xt, xv = np.column_stack(at), np.column_stack(av)
        if not np.isfinite(xt).all() or not np.isfinite(xv).all():
            raise ValueError("Nonfinite V13 feature design")
        return dict(
            xt=xt,
            xv=xv,
            blocks_t=at,
            blocks_v=av,
            widths=[v.shape[1] for v in at],
            labels=labels,
            state=dict(format="v13", context=encoder, microbes=states),
        )

    def microbial(self, site, method, case, iterations, nuisance):
        if method not in ("pc", "sir", "save", "graph"):
            raise ValueError("Unsupported V13 representation")
        a, b = self.view(site, "clr")
        state = self.split.preprocessing["CLR"][site]
        original = np.asarray(state["reference_indices"])[np.asarray(state["output_indices"])]
        if len(original) != a.shape[1]:
            raise ValueError("Frozen CLR/raw taxon mapping differs")
        prevalence = (self.raw[site][np.ix_(self.split.train_rows, original)] > 0).mean(0)
        variable = np.flatnonzero(a.std(0) > 1e-10)
        keep = variable[np.argsort(-prevalence[variable], kind="stable")[: case["count"]]]
        if not len(keep):
            raise ValueError("No variable CLR columns for V13")
        encoder = Directions(method, case.get("components", len(keep)), case.get("slices", 8)).fit(
            a[:, keep], nuisance, self.split.yt
        )
        ct, cv, _ = self.context("joint")
        other = np.column_stack([self.split.hv, cv])
        return (
            encoder.transform(a[:, keep], nuisance),
            encoder.transform(b[:, keep], other),
            dict(
                method=method,
                site=site,
                indices=keep,
                original_indices=original[keep],
                encoder=encoder,
                warnings=[],
            ),
        )

    def replay(self, state, training=False):
        rows, pos = (self.split.train_rows, 0) if training else (self.split.rows, 1)
        blocks = [
            self.split.ht if training else self.split.hv,
            state["context"].transform(self.extra, self.index, rows),
        ]
        nuisance = np.column_stack(blocks)
        for saved in state["microbes"]:
            x = self.view(saved["site"], "clr")[pos][:, saved["indices"]]
            blocks.append(saved["encoder"].transform(x, nuisance))
        return np.column_stack(blocks), blocks
