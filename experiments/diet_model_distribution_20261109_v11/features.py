"""Role-fitted diet distributions, conditional microbial screening and replay."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bridge import LegacyRole, screen
from food_features import DietEncoder


@dataclass
class Role(LegacyRole):
    diet_parameters: object = None

    def context(self, name):
        key = ("context", name)
        if key not in self.cache:
            encoder = DietEncoder(name, self.diet_parameters).fit(
                self.extra, self.index, self.split.train_rows
            )
            self.cache[key] = (
                encoder.transform(self.extra, self.index, self.split.train_rows),
                encoder.transform(self.extra, self.index, self.split.rows),
                encoder,
            )
        return self.cache[key]

    def microbial(self, site, method, case, iterations, nuisance):
        if method not in ("spc", "quadratic"):
            return super().microbial(site, method, case, iterations, nuisance)
        a, b = self.view(site, "clr")
        if method == "spc":
            keep = screen(nuisance, a, self.split.yt, case["count"])
            count = min(case["components"], len(a) - 1, len(keep))
            model = PCA(n_components=count, svd_solver="randomized", random_state=self.seed).fit(a[:, keep])
            at, av = model.transform(a[:, keep]), model.transform(b[:, keep])
            state = dict(site=site, method=method, indices=keep, model=model, warnings=[])
        else:
            scaler = StandardScaler().fit(a)
            z, v = scaler.transform(a), scaler.transform(b)
            residual = self.split.yt - nuisance @ np.linalg.lstsq(nuisance, self.split.yt, rcond=None)[0]
            score = np.zeros(a.shape[1])
            for values in (z, z * z, np.maximum(z, 0)):
                centered = values - values.mean(0)
                score = np.maximum(
                    score, abs(centered.T @ residual) / np.maximum(np.linalg.norm(centered, axis=0), 1e-12)
                )
            k = min(case["count"] or a.shape[1], a.shape[1])
            keep = np.sort(np.argsort(-score, kind="stable")[:k])
            at, av = (
                np.column_stack([z[:, keep], z[:, keep] ** 2]),
                np.column_stack([v[:, keep], v[:, keep] ** 2]),
            )
            state = dict(site=site, method=method, indices=keep, raw_scaler=scaler, warnings=[])
        scaler = StandardScaler().fit(at)
        state["scaler"] = scaler
        return scaler.transform(at), scaler.transform(av), state

    def replay(self, state, training=False):
        rows, pos = (self.split.train_rows, 0) if training else (self.split.rows, 1)
        blocks = [
            self.split.ht if training else self.split.hv,
            state["context"].transform(self.extra, self.index, rows),
        ]
        for saved in state["microbes"]:
            if saved["method"] in ("spc", "quadratic"):
                raw = self.view(saved["site"], "clr")[pos]
                if saved["method"] == "spc":
                    a = saved["model"].transform(raw[:, saved["indices"]])
                else:
                    a = saved["raw_scaler"].transform(raw)[:, saved["indices"]]
                    a = np.column_stack([a, a * a])
                blocks.append(saved["scaler"].transform(a))
            else:
                # The old microbial replay is independent of its context encoder.
                temporary = dict(context=None, microbes=[saved])
                _, parts = super().replay(temporary, training)
                blocks.append(parts[-1])
        return np.column_stack(blocks), blocks

    def modifier_basis(self, source, components):
        if source == "macro":
            from food_features import DietEncoder

            encoder = DietEncoder("joint", self.diet_parameters).fit(
                self.extra, self.index, self.split.train_rows
            )
            a, b = (
                encoder.transform(self.extra, self.index, self.split.train_rows),
                encoder.transform(self.extra, self.index, self.split.rows),
            )
            return a[:, :3], b[:, :3], dict(source="macro", encoder=encoder)
        return super().modifier_basis(source, components)

    def replay_modifier(self, state, training=False):
        if state["source"] == "macro":
            rows = self.split.train_rows if training else self.split.rows
            return state["encoder"].transform(self.extra, self.index, rows)[:, :3]
        return super().replay_modifier(state, training)
