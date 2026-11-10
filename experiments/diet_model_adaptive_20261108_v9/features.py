"""Inductive compositional, zero-aware, ecological and food-variability features."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.special import ndtri
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

from bridge import BaseRole, screen


def closed(x):
    total = x.sum(1, keepdims=True)
    if (total <= 0).any() or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Invalid supported microbial profile")
    return x / total


def ecology(x):
    p = closed(x)
    positive = p > 0
    richness = positive.sum(1)
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1)
    top = np.partition(p, max(0, p.shape[1] - min(5, p.shape[1])), axis=1)[:, -min(5, p.shape[1]) :].sum(1)
    return np.column_stack(
        [
            np.log1p(richness),
            entropy,
            1 - (p * p).sum(1),
            p.max(1),
            top,
            entropy / np.maximum(np.log(np.maximum(richness, 1)), 1e-12),
            1 - richness / p.shape[1],
            np.exp(entropy) / np.maximum(richness, 1),
        ]
    )


def shared_indices(taxa):
    lookup = {v: i for i, v in enumerate(taxa["gut"])}
    pairs = [(i, lookup[v]) for i, v in enumerate(taxa["oral"]) if v in lookup]
    return (np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int))


def cross_ecology(raw, rows, indices):
    a, b = closed(raw["oral"][rows]), closed(raw["gut"][rows])
    i, j = indices
    if not len(i):
        return np.zeros((len(rows), 3))
    x, y = a[:, i], b[:, j]
    union = ((x > 0) | (y > 0)).sum(1)
    return np.column_stack(
        [
            ((x > 0) & (y > 0)).sum(1) / np.maximum(union, 1),
            np.sum(np.sqrt(x * y), axis=1),
            np.sum(x * y, axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12),
        ]
    )


def gate_matrix(index, raw, taxa, rows):
    host = index.iloc[rows][
        ["age_calendar_years", "sex_code", "fiber_density", "mean_daily_energy_fiber"]
    ].to_numpy(float)
    host[:, -1] = np.log(np.maximum(host[:, -1], 1e-12))
    x = np.column_stack(
        [
            host,
            ecology(raw["oral"][rows]),
            ecology(raw["gut"][rows]),
            cross_ecology(raw, rows, shared_indices(taxa)),
        ]
    )
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite gate covariates")
    return x


def rclr(x):
    mask = x > 0
    log = np.log(np.maximum(x, 1e-300))
    center = (log * mask).sum(1, keepdims=True) / np.maximum(mask.sum(1, keepdims=True), 1)
    return (log - center) * mask, mask


def masked_scores(x, loadings, ridge):
    z, mask = rclr(x)
    answer = np.empty((len(x), loadings.shape[1]))
    for start in range(0, len(x), 64):
        m, values = mask[start : start + 64], z[start : start + 64]
        gram = np.einsum("np,pk,pl->nkl", m.astype(float), loadings, loadings, optimize=True)
        gram += ridge * np.eye(loadings.shape[1])[None]
        answer[start : start + 64] = np.linalg.solve(gram, (values @ loadings)[..., None])[..., 0]
    return answer


def completion(x, components, iterations, seed):
    z, mask = rclr(x)
    filled = z.copy()
    k = max(1, min(components, len(x) - 1, x.shape[1]))
    for step in range(iterations):
        u, singular, vt = randomized_svd(filled, n_components=k, n_iter=3, random_state=seed + step)
        low_rank = (u * np.maximum(singular - 0.03 * singular[0], 0)) @ vt
        filled = np.where(mask, z, low_rank)
    return dict(loadings=vt.T, ridge=0.1, iterations=iterations)


def signed_balances(log, host, y, pairs):
    residual = y - host @ np.linalg.lstsq(host, y, rcond=None)[0]
    z = log - log.mean(0)
    score = z.T @ residual / np.maximum(np.linalg.norm(z, axis=0), 1e-12)
    order = np.argsort(-score, kind="stable")
    columns = []
    for k in (1, 2, 4, 8, 16):
        if 2 * k <= len(order):
            w = np.zeros(len(order))
            w[order[:k]], w[order[-k:]] = 1 / k, -1 / k
            columns.append(w * np.sqrt(k / 2))
    width = min(int(np.ceil(np.sqrt(pairs))), len(order) // 2)
    for a in order[:width]:
        for b in order[-width:][::-1]:
            if len(columns) >= pairs + sum(2 * k <= len(order) for k in (1, 2, 4, 8, 16)):
                break
            w = np.zeros(len(order))
            w[a], w[b] = 1 / np.sqrt(2), -1 / np.sqrt(2)
            columns.append(w)
    return np.column_stack(columns)


@dataclass
class Role(BaseRole):
    taxa: object = None
    extra: object = None

    def normal(self, site, rows, columns):
        return closed(self.raw[site][rows])[:, columns]

    def source(self, site):
        key = ("source", site)
        if key not in self.cache:
            train = closed(self.raw[site][self.split.train_rows])
            prevalence = (train > 0).mean(0)
            columns = np.flatnonzero(prevalence >= 0.01)
            if not len(columns):
                raise ValueError("No microbial features meet the training prevalence rule")
            if len(columns) > 2048:
                variance = np.log1p(train[:, columns] * 1e5).var(0)
                columns = np.sort(columns[np.argsort(-variance, kind="stable")[:2048]])
            a, b = train[:, columns], self.normal(site, self.split.rows, columns)
            medians = np.array([np.median(c[c > 0]) for c in a.T])
            self.cache[key] = (a, b, dict(columns=columns, medians=medians))
        return self.cache[key]

    def fit_block(self, site, method, case, iterations):
        s = self.split
        state = dict(kind=method, site=site, warnings=[])
        if method in ("eco", "ecohybrid", "compact"):
            eco_t, eco_v = ecology(self.raw[site][s.train_rows]), ecology(self.raw[site][s.rows])
            a, b = eco_t, eco_v
            if method == "ecohybrid":
                at, av = self.view(site, "clr")
                keep = screen(s.ht, at, s.yt, case["count"])
                state["indices"] = keep
                a, b = np.column_stack([at[:, keep], a]), np.column_stack([av[:, keep], b])
            elif method == "compact":
                at, av, factor = self.factors(site, "pc", case["components"], iterations)
                state.update(factor=factor, warnings=factor["warnings"])
                a, b = np.column_stack([at, a]), np.column_stack([av, b])
        elif method == "rank":
            at, av = self.view(site, "clr")
            keep = screen(s.ht, at, s.yt, case["count"])
            state.update(indices=keep, sorted=np.sort(at[:, keep], axis=0))
            a, b = self.rank(at[:, keep], state["sorted"]), self.rank(av[:, keep], state["sorted"])
        else:
            at, av, source = self.source(site)
            state.update(source)
            if method == "rclr":
                variance = rclr(at)[0].var(0)
                subset = np.sort(np.argsort(-variance, kind="stable")[: min(512, at.shape[1])])
                state["columns"], state["medians"] = state["columns"][subset], state["medians"][subset]
                at, av = at[:, subset], av[:, subset]
                key = ("completion", site, case["components"], iterations)
                if key not in self.cache:
                    self.cache[key] = completion(at, case["components"], iterations, self.seed)
                state["completion"] = self.cache[key]
                a = masked_scores(at, state["completion"]["loadings"], state["completion"]["ridge"])
                b = masked_scores(av, state["completion"]["loadings"], state["completion"]["ridge"])
            else:
                median = state["medians"]
                log_t, log_v = (
                    np.log(np.where(at > 0, at, median * 0.5)),
                    np.log(np.where(av > 0, av, median * 0.5)),
                )
                if method == "hurdle":
                    a = np.column_stack(
                        [at > 0, np.where(at > 0, np.log(np.maximum(at, 1e-300) / median), 0)]
                    )
                    b = np.column_stack(
                        [av > 0, np.where(av > 0, np.log(np.maximum(av, 1e-300) / median), 0)]
                    )
                elif method == "log1p":
                    a, b = np.log1p(at / median), np.log1p(av / median)
                elif method == "alr":
                    spread = log_t.var(0)
                    reference = np.lexsort((spread, -(at > 0).mean(0)))[: min(16, at.shape[1])]
                    state["reference"] = reference
                    a, b = (
                        log_t - log_t[:, reference].mean(1, keepdims=True),
                        log_v - log_v[:, reference].mean(1, keepdims=True),
                    )
                elif method == "balance":
                    keep = screen(s.ht, log_t, s.yt, case["count"])
                    weights = signed_balances(log_t[:, keep], s.ht, s.yt, case["pairs"])
                    state.update(candidates=keep, weights=weights)
                    a, b = log_t[:, keep] @ weights, log_v[:, keep] @ weights
                else:
                    raise ValueError("Unknown V9 representation")
                if method != "balance":
                    keep = screen(s.ht, a, s.yt, case["count"])
                    state["indices"] = keep
                    a, b = a[:, keep], b[:, keep]
        scaler = StandardScaler().fit(a)
        state["scaler"] = scaler
        a, b = scaler.transform(a), scaler.transform(b)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite V9 design")
        return a, b, state

    @staticmethod
    def rank(x, sorted_train):
        n = len(sorted_train)
        out = np.empty_like(x)
        for j in range(x.shape[1]):
            left = np.searchsorted(sorted_train[:, j], x[:, j], side="left")
            right = np.searchsorted(sorted_train[:, j], x[:, j], side="right")
            out[:, j] = ndtri(np.clip((left + right) / (2 * n), 0.5 / n, 1 - 0.5 / n))
        return out

    def design(self, site, method, case, iterations, joint=False):
        if method == "clr":
            return super().design(site, method, case, iterations, False)
        key = ("design", site, method, tuple(sorted(case.items())), joint)
        if key not in self.cache:
            sites = ("oral", "gut") if site == "balanced" else (site,)
            at, av, states = [self.split.ht], [self.split.hv], []
            for name in sites:
                block_key = (
                    "block",
                    name,
                    method,
                    tuple((k, case.get(k)) for k in ("count", "components", "pairs")),
                    iterations,
                )
                if block_key not in self.cache:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        a, b, state = self.fit_block(name, method, case, iterations)
                    state["warnings"] = sorted(
                        set(
                            state["warnings"]
                            + [str(w.message).encode("ascii", "backslashreplace").decode() for w in caught]
                        )
                    )
                    self.cache[block_key] = a, b, state
                    self.cache[("replay", id(state), True)] = state, a
                    self.cache[("replay", id(state), False)] = state, b
                a, b, state = self.cache[block_key]
                at.append(a)
                av.append(b)
                states.append(state)
            if site == "balanced" and method in ("eco", "ecohybrid", "compact"):
                indices = shared_indices(self.taxa)
                a, b = (
                    cross_ecology(self.raw, self.split.train_rows, indices),
                    cross_ecology(self.raw, self.split.rows, indices),
                )
                scaler = StandardScaler().fit(a)
                at.append(scaler.transform(a))
                av.append(scaler.transform(b))
                states.append(dict(kind="cross", indices=indices, scaler=scaler, warnings=[]))
            if joint:
                if self.extra is None:
                    raise ValueError("Daily dietary extension is unavailable")
                a, b = self.extra["values"][self.split.train_rows], self.extra["values"][self.split.rows]
                medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in a.T])

                def fill(x):
                    return np.column_stack([np.where(np.isfinite(x), x, medians), ~np.isfinite(x)])

                scaler = StandardScaler().fit(fill(a))
                at.append(scaler.transform(fill(a)))
                av.append(scaler.transform(fill(b)))
                states.append(dict(kind="daily", medians=medians, scaler=scaler, warnings=[]))
            self.cache[key] = dict(
                xt=np.column_stack(at),
                xv=np.column_stack(av),
                blocks_t=at,
                blocks_v=av,
                state=dict(format="v9", blocks=states),
            )
        return self.cache[key]

    def replay_block(self, state, training):
        rows = self.split.train_rows if training else self.split.rows
        kind, pos = state["kind"], 0 if training else 1
        if kind == "daily":
            x = self.extra["values"][rows]
            a = np.column_stack([np.where(np.isfinite(x), x, state["medians"]), ~np.isfinite(x)])
        elif kind == "cross":
            a = cross_ecology(self.raw, rows, state["indices"])
        elif kind in ("eco", "ecohybrid", "compact"):
            a = ecology(self.raw[state["site"]][rows])
            if kind == "ecohybrid":
                a = np.column_stack([self.view(state["site"], "clr")[pos][:, state["indices"]], a])
            elif kind == "compact":
                f = state["factor"]
                x = self.view(state["site"], "clr")[pos]
                a = np.column_stack([f["scaler"].transform(f["model"].transform(x)), a])
        elif kind == "rank":
            a = self.rank(self.view(state["site"], "clr")[pos][:, state["indices"]], state["sorted"])
        else:
            raw = self.normal(state["site"], rows, state["columns"])
            median = state["medians"]
            log = np.log(np.where(raw > 0, raw, median * 0.5))
            if kind == "hurdle":
                a = np.column_stack(
                    [raw > 0, np.where(raw > 0, np.log(np.maximum(raw, 1e-300) / median), 0)]
                )[:, state["indices"]]
            elif kind == "log1p":
                a = np.log1p(raw / median)[:, state["indices"]]
            elif kind == "alr":
                a = (log - log[:, state["reference"]].mean(1, keepdims=True))[:, state["indices"]]
            elif kind == "balance":
                a = log[:, state["candidates"]] @ state["weights"]
            elif kind == "rclr":
                c = state["completion"]
                a = masked_scores(raw, c["loadings"], c["ridge"])
            else:
                raise ValueError("Unknown replay block")
        return state["scaler"].transform(a)

    def replay(self, states, training=False):
        if states.get("format") != "v9":
            return super().replay(states, training)
        blocks = [self.split.ht if training else self.split.hv]
        for state in states["blocks"]:
            key = ("replay", id(state), training)
            if key not in self.cache:
                self.cache[key] = state, self.replay_block(state, training)
            blocks.append(self.cache[key][1])
        return np.column_stack(blocks), blocks
