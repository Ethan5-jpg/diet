"""Generated-data checks against unprofiled fits and the released V9 controls."""

from copy import copy
import unittest
import numpy as np
import run_v10
from integration_fixture import Generated
from v7_models import _Profile
from test_v7_models import direct_joint
from v8_models import background_and_features
from adapters.fermentation_score import build_score
import v10_grid as g
import v10_features as features
from v10_fit import evaluate


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = Generated(n=180, seed=71265)
        cls.split = cls.src.outer_split(0, 0)
        cls.off, cls.on = evaluate(cls.src, cls.split)

    def test_independent_augmented_least_squares(self):
        s = self.split
        bank, b, doses = features.design_matrices(self.src, s)
        z = (np.column_stack((s.ot, s.gt)), np.column_stack((s.ov, s.gv)))
        for n in (
            g.ANCHOR,
            "score_soft_50_25",
            "dose_rc4__median",
            "panel_omit_rectale",
            "overlay_shannon_gate_6",
            "controlled_both",
            "common_anchor_cell_4",
            *g.CROSS_CELLS["composition"],
            *g.CROSS_CELLS["diversity"],
        ):
            recipe = g.FITS[n]
            x = [
                np.column_stack(
                    [
                        *(bank[m][j] for m in recipe.mains),
                        *(doses[d][j] * bank[m][j] for d, m in recipe.terms),
                    ]
                )
                for j in range(2)
            ]
            m = len(recipe.mains)
            weights = np.r_[np.ones(m), 0.01 * np.array(recipe.weights)]
            for i, expected in (
                (m, self.off[:, g.FAMILY_INDEX[n]]),
                (len(weights), self.on[:, g.INDEX[g.key(n, 0.01)]]),
            ):
                observed = direct_joint(*b, *z, s.yt, x[0][:, :i], x[1][:, :i], weights[:i])
                np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=5e-9, err_msg=n)

    def test_exact_historical_control_at_all_eight_penalties(self):
        s = self.split
        bank, b, d = background_and_features(self.src, s)
        for name in run_v10.REPLAY:
            # Independent of the new global feature matrix, order and profiled product solver.
            m = g.FITS[name].mains
            old = [
                np.column_stack(
                    [*(bank[k][i] for k in m), *(dose * bank[k][i] for dose in d[i] for k in m)]
                )
                for i in (0, 1)
            ]
            model = _Profile(*b, np.column_stack((s.ot, s.gt)), np.column_stack((s.ov, s.gv)), s.yt, *old)
            np.testing.assert_allclose(
                model.prediction(list(range(len(m))), 1),
                self.off[:, g.FAMILY_INDEX[name]],
                atol=1e-10,
                rtol=1e-10,
            )
            for p in g.LEGACY_PENALTIES:
                weights = np.repeat([1.0, p, 3 * p], len(m))
                np.testing.assert_allclose(
                    model.prediction(list(range(len(weights))), weights),
                    self.on[:, g.INDEX[g.key(name, p)]],
                    atol=1e-9,
                    rtol=1e-10,
                )

    def test_fold_safe_predictions_after_replacing_validation_data(self):
        t = copy(self.split)
        t.yv = np.full_like(self.split.yv, np.nan)
        a, b = evaluate(self.src, t)
        np.testing.assert_array_equal(a, self.off)
        np.testing.assert_array_equal(b, self.on)
        # Perturb one validation participant's microbiome and fiber; every other prediction stays fixed.
        other = copy(self.src)
        other.raw = {k: x.copy() for k, x in self.src.raw.items()}
        idx = t.rows[0]
        other.raw["gut"][idx] = np.roll(other.raw["gut"][idx], 3)
        t.dv = t.dv.copy()
        t.dv[0, :] += 1.0
        t.hv = t.hv.copy()
        t.hv[0, :2] += 1.0
        t.gv = t.gv.copy()
        t.gv[0, :] += 0.2
        a, b = evaluate(other, t)
        np.testing.assert_allclose(a[1:], self.off[1:], atol=1e-9, rtol=0)
        np.testing.assert_allclose(b[1:], self.on[1:], atol=1e-9, rtol=0)

    def test_marginal_hierarchy_and_paired_ensembles(self):
        for name, r in g.FITS.items():
            for d, m in r.terms:
                self.assertTrue(m in r.mains or m in {"g1", "g2", "g3", "g4"}, name)
                self.assertTrue(d in (g.U, g.C) or d in r.mains, name)
                for gate, parent in (
                    ("shannon_x_fermentation", g.S),
                    ("f_gate_g1", "g1"),
                    ("f_gate_g2", "g2"),
                ):
                    if m == gate:
                        self.assertIn((d, parent), r.terms)
                        self.assertIn((d, g.F), r.terms)
        for n, parents in g.BLENDS.items():
            np.testing.assert_allclose(
                self.off[:, g.FAMILY_INDEX[n]],
                sum(w * self.off[:, g.FAMILY_INDEX[parent]] for parent, w in parents),
            )
            for p in g.SETTINGS[n]:
                np.testing.assert_allclose(
                    self.on[:, g.INDEX[g.key(n, p)]],
                    sum(w * self.on[:, g.INDEX[g.key(parent, p)]] for parent, w in parents),
                )
        for n, (parent, w) in g.FRACTIONS.items():
            base = self.off[:, g.FAMILY_INDEX[parent]]
            np.testing.assert_array_equal(self.off[:, g.FAMILY_INDEX[n]], base)
            for p in g.SETTINGS[n]:
                i = g.INDEX[g.key(n, p)]
                np.testing.assert_allclose(
                    self.on[:, i], base + w * (self.on[:, g.INDEX[g.key(parent, p)]] - base)
                )
        for n, (_, parent) in g.ALT.items():
            np.testing.assert_array_equal(
                self.off[:, g.FAMILY_INDEX[n]], self.off[:, g.FAMILY_INDEX[parent]]
            )
        for names in (g.CELLS, [n for n in g.FITS if n.startswith("controlled_")], *g.CROSS_CELLS.values()):
            for n in names:
                np.testing.assert_allclose(
                    self.off[:, g.FAMILY_INDEX[n]],
                    self.off[:, g.FAMILY_INDEX[names[0]]],
                    rtol=0,
                    atol=2e-10,
                )

    def test_every_fixed_settings_null_against_raw_augmented_regression(self):
        from v10_fit import audit_matched_off

        s = self.split
        bank, bases, _ = features.design_matrices(self.src, s)
        z = np.column_stack((s.ot, s.gt)), np.column_stack((s.ov, s.gv))
        # No product column is even constructed in the independent raw null oracle.
        raw = {}
        for mains in {rec.mains for rec in g.FITS.values()}:
            x = [
                np.column_stack([bank[m][side] for m in mains]) if mains else np.empty((len(b), 0))
                for side, b in enumerate(bases)
            ]
            raw[mains] = direct_joint(*bases, *z, s.yt, *x, np.ones(len(mains)))

        def expected(n):
            if n in g.FITS:
                return raw[g.FITS[n].mains]
            if n in g.BLENDS:
                return sum(w * expected(p) for p, w in g.BLENDS[n])
            parent = g.ALT[n][1] if n in g.ALT else g.FRACTIONS[n][0]
            return expected(parent)

        truth = np.column_stack([expected(n) for n in g.FAMILIES])
        by_name = {n: j for j, n in enumerate(g.FAMILIES)}
        # Independently parse the labels instead of deriving the expected column with OFF_INDEX.
        oracle = truth[:, [by_name[k.rsplit("@", 1)[0]] for k in g.KEYS]]
        np.testing.assert_allclose(self.off[:, g.OFF_INDEX], oracle, rtol=0, atol=1e-8)
        corrupt = self.off.copy()
        i, j = g.FAMILY_INDEX[g.ANCHOR], g.FAMILY_INDEX["only_balances"]
        self.assertGreater(np.max(np.abs(corrupt[:, i] - corrupt[:, j])), 1e-5)
        corrupt[:, [i, j]] = corrupt[:, [j, i]]
        with self.assertRaisesRegex(ValueError, "main-only fit"):
            audit_matched_off(corrupt, raw.__getitem__)

    def test_species_resolution_composition_and_true_zeros(self):
        s = self.split
        raw = self.src.raw["gut"]
        alias = list(self.src.taxa["gut"])
        alias[2] = "s__Agathobacter_rectalis"
        expected, _, panel = build_score(raw, alias)
        np.testing.assert_allclose(expected, np.log1p(panel.sum(axis=1) / 0.0001))
        for invalid in (alias[1:], alias + [alias[0]]):
            with self.assertRaises(ValueError):
                features.resolve_species(invalid)
        other = copy(self.src)
        other.raw = {k: a.copy() for k, a in self.src.raw.items()}
        other.raw["gut"][s.rows[0], :3] = 0.0
        bank = {g.F: (np.zeros(len(s.train_rows)), np.zeros(len(s.rows)))}
        for j in (1, 2):
            bank[f"g{j}"] = bank[g.F]
        features.add_composition(other, s, bank)
        for k in ("panel_dominance", "panel_evenness", "omit_rectale", "detect_any"):
            self.assertTrue(all(np.isfinite(x).all() for x in bank[k]))
        p = other.raw["gut"][:, :3]
        v = np.log1p(p[:, :2].sum(axis=1) / 0.0001)
        np.testing.assert_allclose(bank["omit_rectale"][1], v[s.rows] - v[s.train_rows].mean())
        np.testing.assert_allclose(bank["detect_any"][1][0], -1.0)

    def test_spline_ties_and_linear_extrapolation(self):
        a = np.linspace(-2, 2, 101)
        for train in (a, np.zeros_like(a), np.r_[np.zeros(95), np.ones(6)]):
            for x, y in features.rc_basis((train, np.array([4.0, 5.0, 6.0])), [10, 35, 65, 90]):
                self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())
                self.assertAlmostEqual(y[2] - y[1], y[1] - y[0], places=8)
        bank, _, dose = features.design_matrices(self.src, self.split)
        for k, pair in dose.items():
            if k not in (g.U, g.C):
                self.assertAlmostEqual(pair[0].mean(), 0.0, places=9)
                self.assertAlmostEqual(pair[0] @ dose[g.U][0], 0.0, places=8)

    def test_positive_and_additive_generating_processes(self):
        src = self.src
        score, _, _ = build_score(src.raw["gut"], src.taxa["gut"])
        u = (src.fiber - 13.0) / 4.0
        main = 80 + 0.25 * src.age + 0.75 * src.fiber
        for effect in (0.0, 8.0):
            data = src.with_y(main + effect * (score - score.mean()) * u)
            fold = data.outer_split(0, 0)
            off, on = evaluate(data, fold)
            base = off[:, g.FAMILY_INDEX[g.ANCHOR]]
            full = on[:, g.INDEX[g.REFERENCE]]
            if effect:
                self.assertLess(np.sum((full - fold.yv) ** 2), np.sum((base - fold.yv) ** 2) * 0.25)
            else:
                np.testing.assert_allclose(base, fold.yv, rtol=0, atol=1e-8)
                np.testing.assert_allclose(full, base, rtol=0, atol=1e-8)
