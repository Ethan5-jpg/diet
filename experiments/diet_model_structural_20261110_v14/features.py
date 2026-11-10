"""Frozen CLR mapping, conditional projections and training/validation replay."""

from dataclasses import dataclass

import numpy as np

from bridge import V11Role, SIRDirections
from projections import NonlinearDirections, PairedViews


@dataclass
class Role(V11Role):
    def selected(self, site, count):
        a, b = self.view(site, "clr")
        state = self.split.preprocessing["CLR"][site]
        original = np.asarray(state["reference_indices"])[np.asarray(state["output_indices"])]
        if len(original) != a.shape[1]:
            raise ValueError("Frozen CLR/raw taxon mapping differs")
        prevalence = (self.raw[site][np.ix_(self.split.train_rows, original)] > 0).mean(0)
        variable = np.flatnonzero(a.std(0) > 1e-10)
        keep = variable[np.argsort(-prevalence[variable], kind="stable")[:count]]
        if not len(keep):
            raise ValueError("No variable CLR columns")
        return a[:, keep], b[:, keep], dict(site=site, indices=keep, original_indices=original[keep])

    def design(self, site, method, case, iterations, context="joint"):
        ct, cv, encoder = self.context(context)
        nt, nv = np.column_stack([self.split.ht, ct]), np.column_stack([self.split.hv, cv])
        key = ("representation", site, method) + tuple(
            case[k]
            for k in (
                "count",
                "components",
                "slices",
                "pre_components",
                "rff_features",
                "bandwidth",
                "pilot_penalty",
                "cca_penalty",
            )
        )
        if key not in self.cache:
            sites = ("oral", "gut") if site == "balanced" else (site,)
            selected = [self.selected(s, case["count"]) for s in sites]
            train, valid, states = [], [], []
            if method in ("cca", "paired_pc"):
                if site != "balanced":
                    raise ValueError("Paired projections require both sites")
                model = PairedViews(method, case, self.seed).fit([v[0] for v in selected], nt)
                train.append(model.transform([v[0] for v in selected], nt))
                valid.append(model.transform([v[1] for v in selected], nv))
                states.append(dict(paired=True, inputs=[v[2] for v in selected], encoder=model))
            else:
                for a, b, state in selected:
                    model = (
                        SIRDirections("sir", case["components"], case["slices"])
                        if method == "sir"
                        else NonlinearDirections(method, case, self.seed)
                    )
                    model.fit(a, nt, self.split.yt)
                    train.append(model.transform(a, nt))
                    valid.append(model.transform(b, nv))
                    states.append(dict(state, paired=False, encoder=model))
            self.cache[key] = train, valid, states
        a, b, states = self.cache[key]
        at, av = [self.split.ht, ct] + a, [self.split.hv, cv] + b
        xt, xv = np.column_stack(at), np.column_stack(av)
        if not np.isfinite(xt).all() or not np.isfinite(xv).all():
            raise ValueError("Nonfinite V14 feature design")
        return dict(
            xt=xt,
            xv=xv,
            widths=[x.shape[1] for x in at],
            state=dict(format="v14", context=encoder, microbes=states),
        )

    def replay(self, state, training=False):
        rows, pos = (self.split.train_rows, 0) if training else (self.split.rows, 1)
        blocks = [
            self.split.ht if training else self.split.hv,
            state["context"].transform(self.extra, self.index, rows),
        ]
        nuisance = np.column_stack(blocks)
        for saved in state["microbes"]:
            if saved["paired"]:
                x = [self.view(s["site"], "clr")[pos][:, s["indices"]] for s in saved["inputs"]]
            else:
                x = self.view(saved["site"], "clr")[pos][:, saved["indices"]]
            blocks.append(saved["encoder"].transform(x, nuisance))
        return np.column_stack(blocks), blocks
