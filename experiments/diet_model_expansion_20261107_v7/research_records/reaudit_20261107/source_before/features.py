"""Train-role-only representations, balanced site screening and cached designs."""

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA


def selected(host, microbes, target, count):
    if count == 0 or count >= microbes.shape[1]:
        return np.arange(microbes.shape[1])
    residual = target - host @ np.linalg.lstsq(host, target, rcond=None)[0]
    centered = microbes - microbes.mean(0)
    norm = np.sqrt(np.sum(centered**2, axis=0))
    score = np.divide(abs(centered.T @ residual), norm, out=np.zeros_like(norm), where=norm > 1e-12)
    return np.sort(np.argsort(-score, kind="stable")[:count])


@dataclass
class Design:
    xt: np.ndarray
    xv: np.ndarray
    blocks_t: list
    blocks_v: list
    gut_columns: list
    state: dict


@dataclass
class Role:
    split: object
    raw: dict
    seed: int
    cache: dict = field(default_factory=dict)

    def view(self, site, representation):
        key = ("view", site, representation)
        if key not in self.cache:
            s = self.split
            if representation == "clr":
                train, valid = (s.ot, s.ov) if site == "oral" else (s.gt, s.gv)
            else:
                state = s.preprocessing["CLR"][site]
                ref, output = state["reference_indices"], state["output_indices"]
                train = self.raw[site][np.ix_(s.train_rows, ref)]
                valid = self.raw[site][np.ix_(s.rows, ref)]

                def transform(a):
                    total = a.sum(1, keepdims=True)
                    if (total <= 0).any():
                        raise ValueError("Unsupported profile in matched eligible rows")
                    if representation == "hellinger":
                        return np.sqrt(a / total)[:, output]
                    if representation == "presence":
                        return (a[:, output] > 0).astype(float)
                    raise ValueError("Unknown microbial representation")

                train, valid = transform(train), transform(valid)
            if not np.isfinite(train).all() or not np.isfinite(valid).all():
                raise ValueError("Nonfinite role features")
            self.cache[key] = train, valid
        return self.cache[key]

    def pcs(self, site="gut", count=10):
        key = ("pc", site, count)
        if key not in self.cache:
            train, valid = self.view(site, "clr")
            count = min(count, train.shape[1], len(train) - 1)
            model = PCA(n_components=count, svd_solver="randomized", random_state=self.seed)
            a, b = model.fit_transform(train), model.transform(valid)
            scale = a.std(0)
            scale[scale < 1e-12] = 1.0
            self.cache[key] = (a / scale, b / scale, model, scale)
        return self.cache[key]

    def design(self, site, representation="clr", count=64, hybrid=False):
        key = ("design", site, representation, count, hybrid)
        if key not in self.cache:
            s = self.split
            blocks_t, blocks_v = [s.ht], [s.hv]
            states, gut_columns, offset = {}, [], s.ht.shape[1]
            sites = [] if site == "host" else (["oral", "gut"] if site == "balanced" else [site])
            for name in sites:
                a, b = self.view(name, representation)
                indices = selected(s.ht, a, s.yt, count)
                a, b = a[:, indices], b[:, indices]
                states[name] = dict(indices=indices, representation=representation)
                if hybrid and name == "gut":
                    pc_t, pc_v, pc, scale = self.pcs()
                    a, b = np.column_stack([a, pc_t]), np.column_stack([b, pc_v])
                    states[name].update(pc=pc, pc_scale=scale)
                if name == "gut":
                    gut_columns = list(range(offset, offset + len(indices)))
                blocks_t.append(a)
                blocks_v.append(b)
                offset += a.shape[1]
            self.cache[key] = Design(
                np.column_stack(blocks_t),
                np.column_stack(blocks_v),
                blocks_t,
                blocks_v,
                gut_columns,
                states,
            )
        return self.cache[key]

    def low_high(self, matrix, contrast):
        result = []
        state = self.split.preprocessing
        j = state["host"]["labels"].index("diet_linear")
        for value in contrast:
            x = matrix.copy()
            x[:, j] = (value - state["diet"]["center"]) / state["diet"]["scale"]
            result.append(x)
        return result

    def replay(self, state, training=False):
        position = 0 if training else 1
        blocks = [self.split.ht if training else self.split.hv]
        for site, saved in state.items():
            x = self.view(site, saved["representation"])[position][:, saved["indices"]]
            if "pc" in saved:
                clr = self.view(site, "clr")[position]
                x = np.column_stack([x, saved["pc"].transform(clr) / saved["pc_scale"]])
            blocks.append(x)
        return np.column_stack(blocks), blocks
