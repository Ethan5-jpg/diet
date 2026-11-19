"""Training-only microbiome CLR and endpoint outlier transforms; no data I/O.

Inputs must already satisfy identity, source-null semantics and basic QC.
Real-data callers must use the project access guard and role registry.
Serialization contains fitted parameters, never participant IDs or raw rows.
"""
import json
from pathlib import Path

import numpy as np


def _training_units(ids, n, role):
    if role != "training":
        raise ValueError("fit requires training role; validation is transform-only")
    ids = list(ids)
    if len(ids) != n or any(x is None or not str(x).strip() for x in ids):
        raise ValueError("one nonmissing participant ID is required per fit row")
    if len(set(map(str, ids))) != n:
        raise ValueError("fit rows must be unique participants, not repeated days")


class SavedTransform:
    kind = None

    def save(self, path):
        Path(path).write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.state = json.loads(Path(path).read_text())
        if obj.state.get("kind") != cls.kind:
            raise ValueError("incorrect transform kind")
        return obj


class MicrobiomeCLR(SavedTransform):
    kind = "hpp_species_clr_5pct_v1"

    @staticmethod
    def _matrix(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("nonempty full-species matrix required")
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("resolve source missingness/invalid abundances before CLR")
        if not np.allclose(x.sum(axis=1), 1.0, rtol=0, atol=1e-3):
            raise ValueError("expected full profiles as fractions summing to one")
        return x

    @staticmethod
    def _clr(x, pseudocounts):
        a = np.where(x == 0, pseudocounts, x)
        a = a / a.sum(axis=1, keepdims=True)
        logs = np.log(a)
        return logs - logs.mean(axis=1, keepdims=True)

    def fit(self, x, feature_names, participant_ids, *, site, role="training"):
        x = self._matrix(x)
        _training_units(participant_ids, len(x), role)
        names = list(feature_names)
        if len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError("unique feature names required in source order")
        if site not in ("oral", "gut"):
            raise ValueError("fit oral and gut separately")
        prevalence = (x > 0).mean(axis=0)
        keep = np.flatnonzero(prevalence >= 0.05)
        if len(keep) < 2:
            raise ValueError("insufficient species support for CLR")
        selected = x[:, keep]
        pseudo = np.array([
            0.5 * np.quantile(a[a > 0], 0.01, method="linear")
            for a in selected.T
        ])
        if not np.isfinite(pseudo).all() or (pseudo <= 0).any():
            raise ValueError("invalid training pseudocount")
        supported = selected.sum(axis=1) > 0
        clr = self._clr(selected[supported], pseudo)
        means, scales = clr.mean(axis=0), clr.std(axis=0, ddof=0)
        output = np.flatnonzero(scales > 1e-12)
        if len(output) == 0:
            raise ValueError("no informative CLR columns; no score supported")
        self.state = {
            "kind": self.kind, "site": site, "training_participants": len(x),
            "training_supported_participants": int(supported.sum()),
            "source_features": names, "reference_indices": keep.tolist(),
            "prevalence": prevalence[keep].tolist(), "pseudocounts": pseudo.tolist(),
            "output_indices": output.tolist(), "means": means[output].tolist(),
            "scales": scales[output].tolist(),
        }
        return self

    def transform(self, x, feature_names, *, site):
        x = self._matrix(x)
        s = self.state
        if list(feature_names) != s["source_features"] or site != s["site"]:
            raise ValueError("source taxon order/site differs from fitted transform")
        selected = x[:, s["reference_indices"]]
        # All-zero retained taxa indicate a profile unsupported by this reference.
        # Do not fabricate a microbiome from pseudocounts alone.
        supported = selected.sum(axis=1) > 0
        z = np.full((len(x), len(s["output_indices"])), np.nan)
        if supported.any():
            clr = self._clr(selected[supported], np.asarray(s["pseudocounts"]))
            z[supported] = (clr[:, s["output_indices"]] - s["means"]) / s["scales"]
        names = [s["source_features"][s["reference_indices"][j]] for j in s["output_indices"]]
        return {"values": z, "feature_names": names, "supported": supported}


class PhenotypeOutliers(SavedTransform):
    kind = "hpp_endpoint_shortest95_8sd_5sd_v1"

    def fit(self, y, participant_ids, *, endpoint, role="training"):
        a = np.asarray(y, dtype=float)
        if a.ndim != 1:
            raise ValueError("one endpoint at a time")
        _training_units(participant_ids, len(a), role)
        ordered = np.sort(a[np.isfinite(a)])
        n = len(ordered)
        if n < 3:
            raise ValueError("insufficient finite training observations")
        k = int(np.ceil(0.95 * n))
        widths = ordered[k-1:] - ordered[:n-k+1]
        start = int(np.argmin(widths))
        core = ordered[start:start+k]
        mu, sd = float(core.mean()), float(core.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError("degenerate robust SD; endpoint needs support review")
        self.state = {
            "kind": self.kind, "endpoint": endpoint, "training_units": len(a),
            "training_finite": n, "interval_n": k,
            "interval_low": float(core[0]), "interval_high": float(core[-1]),
            "mean": mu, "sd": sd,
            "remove_low": mu - 8*sd, "remove_high": mu + 8*sd,
            "clip_low": mu - 5*sd, "clip_high": mu + 5*sd,
        }
        return self

    def transform(self, y, *, endpoint):
        a = np.asarray(y, dtype=float)
        if a.ndim != 1 or endpoint != self.state["endpoint"]:
            raise ValueError("endpoint/shape mismatch")
        s = self.state
        finite = np.isfinite(a)
        extreme = finite & ((a < s["remove_low"]) | (a > s["remove_high"]))
        usable = finite & ~extreme
        low = usable & (a < s["clip_low"])
        high = usable & (a > s["clip_high"])
        cleaned = np.full(a.shape, np.nan)
        cleaned[usable] = np.clip(a[usable], s["clip_low"], s["clip_high"])
        return {"original": a.copy(), "values": cleaned,
                "nonfinite": ~finite, "excluded_8sd": extreme,
                "winsorized_low": low, "winsorized_high": high}
