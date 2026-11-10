"""Replayable, training-role context encoding and microbial block construction."""

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from bridge import BaseRole, screen

LIFESTYLE_BLOCKS = ("life", "full", "macro_life", "joint", "fiber_days", "distribution", "ratios")


@dataclass
class ContextEncoder:
    name: str

    def fit(self, extra, rows):
        a = extra["blocks"][self.name][rows]
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in a.T])
        self.categories = (
            sorted({v for v in extra["alcohol"][rows] if v is not None})
            if self.name in LIFESTYLE_BLOCKS
            else []
        )
        self.scaler = StandardScaler().fit(self.raw(extra, rows))
        return self

    def raw(self, extra, rows):
        a = extra["blocks"][self.name][rows]
        parts = [np.where(np.isfinite(a), a, self.medians), ~np.isfinite(a)]
        if self.name in LIFESTYLE_BLOCKS:
            cat = extra["alcohol"][rows]
            parts += [
                np.column_stack([cat == v for v in self.categories])
                if self.categories
                else np.empty((len(rows), 0)),
                (cat == None)[:, None],  # noqa: E711
                np.array([v is not None and v not in self.categories for v in cat])[:, None],
            ]
        return np.column_stack(parts).astype(float)

    def transform(self, extra, rows):
        return self.scaler.transform(self.raw(extra, rows))


def closed(raw):
    total = raw.sum(1, keepdims=True)
    if (total <= 0).any() or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid supported microbial profile")
    return raw / total


def ecology(raw):
    p = closed(raw)
    rich = (p > 0).sum(1)
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
    return np.column_stack(
        [
            np.log1p(rich),
            entropy,
            1 - (p * p).sum(1),
            p.max(1),
            entropy / np.maximum(np.log(np.maximum(rich, 1)), 1e-12),
        ]
    )


@dataclass
class Role(BaseRole):
    taxa: object = None
    extra: object = None

    def context(self, name):
        if name == "none":
            return np.empty((len(self.split.yt), 0)), np.empty((len(self.split.yv), 0)), None
        key = ("context", name)
        if key not in self.cache:
            encoder = ContextEncoder(name).fit(self.extra, self.split.train_rows)
            self.cache[key] = (
                encoder.transform(self.extra, self.split.train_rows),
                encoder.transform(self.extra, self.split.rows),
                encoder,
            )
        return self.cache[key]

    def microbial(self, site, method, case, iterations, nuisance):
        s = self.split
        state = dict(site=site, method=method, warnings=[])
        if method == "pc":
            a, b, factor = self.factors(site, "pc", case["components"], iterations)
            return a, b, dict(state, factor=factor, warnings=factor["warnings"])
        if method == "hurdle":
            train, valid = closed(self.raw[site][s.train_rows]), closed(self.raw[site][s.rows])
            columns = np.flatnonzero((train > 0).mean(0) >= 0.01)
            if not len(columns):
                raise ValueError("No training taxa meet prevalence rule")
            train, valid = train[:, columns], valid[:, columns]
            median = np.array([np.median(c[c > 0]) for c in train.T])
            state.update(columns=columns, median=median)

            def transform(x):
                return np.column_stack([x > 0, np.where(x > 0, np.log(np.maximum(x, 1e-300) / median), 0)])

            a, b = transform(train), transform(valid)
        else:
            a, b = self.view(site, "clr")
        keep = screen(nuisance, a, s.yt, case.get("count", 0))
        state["indices"] = keep
        a, b = a[:, keep], b[:, keep]
        if method == "ecohybrid":
            a = np.column_stack([a, ecology(self.raw[site][s.train_rows])])
            b = np.column_stack([b, ecology(self.raw[site][s.rows])])
        scaler = StandardScaler().fit(a)
        state["scaler"] = scaler
        return scaler.transform(a), scaler.transform(b), state

    def design(self, site, method, case, iterations, context="none"):
        key = ("design", site, method, tuple(sorted(case.items())), context)
        if key not in self.cache:
            ct, cv, encoder = self.context(context)
            at, av = [self.split.ht, ct], [self.split.hv, cv]
            states, labels = [], ["host", "context"]
            sites = ("oral", "gut") if site == "balanced" else (() if site == "host" else (site,))
            for name in sites:
                block_key = (
                    "block",
                    name,
                    method,
                    case.get("count", 0),
                    case.get("components", 0),
                    iterations,
                    context,
                )
                if block_key not in self.cache:
                    self.cache[block_key] = self.microbial(
                        name, method, case, iterations, np.column_stack(at[:2])
                    )
                a, b, state = self.cache[block_key]
                at.append(a)
                av.append(b)
                states.append(state)
                labels.append(name)
            xt, xv = np.column_stack(at), np.column_stack(av)
            if not np.isfinite(xt).all() or not np.isfinite(xv).all():
                raise ValueError("Nonfinite V10 feature design")
            self.cache[key] = dict(
                xt=xt,
                xv=xv,
                blocks_t=at,
                blocks_v=av,
                widths=[v.shape[1] for v in at],
                labels=labels,
                state=dict(format="v10", context=encoder, microbes=states),
            )
        return self.cache[key]

    def replay(self, state, training=False):
        rows, pos = (self.split.train_rows, 0) if training else (self.split.rows, 1)
        blocks = [self.split.ht if training else self.split.hv]
        encoder = state["context"]
        blocks.append(
            encoder.transform(self.extra, rows) if encoder is not None else np.empty((len(rows), 0))
        )
        for saved in state["microbes"]:
            site, method = saved["site"], saved["method"]
            if method == "pc":
                f = saved["factor"]
                a = f["scaler"].transform(f["model"].transform(self.view(site, "clr")[pos]))
            else:
                if method == "hurdle":
                    p = closed(self.raw[site][rows])[:, saved["columns"]]
                    a = np.column_stack(
                        [p > 0, np.where(p > 0, np.log(np.maximum(p, 1e-300) / saved["median"]), 0)]
                    )
                else:
                    a = self.view(site, "clr")[pos]
                a = a[:, saved["indices"]]
                if method == "ecohybrid":
                    a = np.column_stack([a, ecology(self.raw[site][rows])])
                a = saved["scaler"].transform(a)
            blocks.append(a)
        return np.column_stack(blocks), blocks

    def modifier_basis(self, source, components):
        if source == "macro":
            a, b, encoder = self.context("macro")
            return a[:, :3], b[:, :3], dict(source="macro", encoder=encoder)
        states, train, valid = [], [], []
        for site in ("oral", "gut") if source == "balanced" else (source,):
            a, b, state = self.factors(site, "pc", components, 1)
            train.append(a)
            valid.append(b)
            states.append(dict(site=site, factor=state))
        return np.column_stack(train), np.column_stack(valid), dict(source=source, factors=states)

    def replay_modifier(self, state, training=False):
        rows, pos = (self.split.train_rows, 0) if training else (self.split.rows, 1)
        if state["source"] == "macro":
            return state["encoder"].transform(self.extra, rows)[:, :3]
        return np.column_stack(
            [
                s["factor"]["scaler"].transform(
                    s["factor"]["model"].transform(self.view(s["site"], "clr")[pos])
                )
                for s in state["factors"]
            ]
        )
