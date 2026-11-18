"""Independent numerical and role-isolation checks for V7 paired models."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

# Standalone unittest and the server entry point use the installed V6 fixture.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v6"))
from integration_fixture import Generated
import v7_models as m


def direct_joint(b, bv, z, zv, y, w, wv, weights):
    """Solve the raw augmented least-squares objective without profiling."""
    n, q = z.shape
    sz = np.where(z.std(0) > 1e-12, z.std(0), 1)
    sw = np.where(w.std(0) > 1e-12, w.std(0), 1)
    a = np.column_stack((b, z / sz, w / sw))
    av = np.column_stack((bv, zv / sz, wv / sw))
    penalty = np.r_[np.zeros(b.shape[1]), np.ones(q), weights]
    reg = np.diag(np.sqrt(n * penalty))
    beta = linalg.lstsq(np.vstack((a, reg)), np.r_[y, np.zeros(len(penalty))],
                        cond=1e-11, lapack_driver="gelsd")[0]
    return av @ beta


class PairedModelTests(unittest.TestCase):
    def test_dual_and_primal_profile_match_joint_augmented_least_squares(self):
        rng = np.random.default_rng(1024)
        for n, width in [(90, 20), (30, 50)]:
            b = np.column_stack((np.ones(n), rng.normal(size=(n, 4))))
            b = np.column_stack((b, b[:, 1]))  # Force a dependent control.
            bv = np.column_stack((np.ones(7), rng.normal(size=(7, 4))))
            bv = np.column_stack((bv, bv[:, 1]))
            z, zv = rng.normal(size=(n, width)), rng.normal(size=(7, width))
            w, wv = rng.normal(size=(n, 7)), rng.normal(size=(7, 7))
            y = rng.normal(size=n)
            profiled = m._Profile(b, bv, z, zv, y, w, wv)
            for indices in ([], [3, 0, 2, 4]):
                penalty = .1 * np.arange(1, 1 + len(indices))
                expected = direct_joint(b, bv, z, zv, y, w[:, indices], wv[:, indices], penalty)
                np.testing.assert_allclose(profiled.prediction(indices, penalty), expected,
                                           rtol=1e-9, atol=1e-10)

    def test_anchor_and_nonlinear_marginal_hierarchy_vs_direct_fits(self):
        source = Generated()
        split = source.outer_split(0, 0)
        with threadpool_limits(limits=2):
            predictions = m.evaluate_split(source, split)
            bank = m._feature_bank(source, split)
            b, bv = [np.column_stack((np.ones(len(h)), h, dose, *(bank[p][i]
                       for p in (*m.pc_list("o", 5), *m.pc_list("g", 5)))))
                     for i, (h, dose) in enumerate(((split.ht, split.dt), (split.hv, split.dv)))]
            z, zv = np.column_stack((split.ot, split.gt)), np.column_stack((split.ov, split.gv))
            for family in (m.ANCHOR, "gut_pc10_hinge", "gut_pc10_shannon", "gut_pc10_dose_spline",
                           "gut_pc10_marker_gate_both", "gut_pc10_plus_rff_broad",
                           "gut_taxa_pivot", "gut_pc10_plus_taxa_detect_intensity"):
                route = next(r for r in m.ROUTES if r.name == family)
                extras = [p for p in route.modifiers if m._extra_name(p)]
                w = [np.column_stack([*(bank[p][i] for p in extras),
                                      *(dose[:, d] * bank[p][i] for d in route.doses for p in route.modifiers)])
                     for i, dose in enumerate((split.dt, split.dv))]
                weights = np.r_[np.ones(len(extras)), np.full(len(route.modifiers)*len(route.doses), .1)]
                for arm, end in enumerate([len(extras), w[0].shape[1]]):
                    expected = direct_joint(b, bv, z, zv, split.yt, w[0][:, :end], w[1][:, :end], weights[:end])
                    np.testing.assert_allclose(predictions[m.setting_key(family, .1)][arm], expected,
                                               rtol=1e-9, atol=1e-9)

    def test_paired_blends_both_arms_and_within_family_nulls(self):
        source = Generated()
        with threadpool_limits(limits=2):
            predictions = m.evaluate_split(source, source.outer_split(0, 0))
        self.assertEqual(list(predictions), [k for keys in m.families().values() for k in keys])
        self.assertEqual(len(predictions), 260)
        for keys in m.families().values():
            for key in keys:
                np.testing.assert_array_equal(predictions[key][0], predictions[keys[0]][0])
        for name, members in m.BLENDS.items():
            for penalty in m.PENALTIES:
                for arm in (0, 1):
                    expected = sum(predictions[m.setting_key(p, penalty)][arm] for p in members)/len(members)
                    np.testing.assert_allclose(predictions[m.setting_key(name, penalty)][arm], expected, atol=1e-11)
        self.assertGreater(np.std(predictions["gut_pc10_fermentation@0.1"][0] -
                                  predictions[m.FIXED_ANCHOR_SETTING][0]), 1e-5)

    def test_outer_audit_and_ecological_products_follow_training_role(self):
        source = Generated()
        split = source.outer_split(0, 0)
        audit = {}
        with threadpool_limits(limits=2):
            bank = m._feature_bank(source, split, view_audit=audit)
        self.assertEqual(audit["training_rows"], len(split.train_rows))
        indices = audit["selected_gut_indices"]
        self.assertEqual(len(set(indices)), m.views.KEEP_TAXA)
        self.assertEqual(audit["selected_gut_taxa"], [source.taxa["gut"][j] for j in indices])
        prevalence = (source.raw["gut"][split.train_rows][:, indices] > 0).mean(axis=0)
        np.testing.assert_array_equal(audit["selected_training_prevalences"], prevalence)
        for marker in ("shannon", "fermentation"):
            for i in range(1, 5):
                pc = bank[f"g{i}"]
                raw_product = [(part - pc[0].mean()) * marker_part
                               for part, marker_part in zip(pc, bank[marker])]
                for actual, raw in zip(bank[f"v7_gate_{marker}_g{i}"], raw_product):
                    np.testing.assert_allclose(actual, raw - raw_product[0].mean(), atol=1e-13)

    def test_validation_targets_and_other_validation_people_do_not_affect_predictions(self):
        source = Generated()
        split = source.outer_split(0, 0)
        with threadpool_limits(limits=2):
            predictions = m.evaluate_split(source, split)
            # Poison validation targets, and alter just the first person's diet
            # and microbiome features. Other validation rows must be unchanged.
            split.yv[:] = np.nan
            split.hv[0, :] += .5
            split.dv[0, :] += .5
            split.gv[0, :] += .5
            split.ov[0, :] += .5
            replay = m.evaluate_split(source, split)
        for key in predictions:
            for before, after in zip(predictions[key], replay[key]):
                np.testing.assert_allclose(before[1:], after[1:], rtol=1e-12, atol=1e-12)

    def test_pca_center_and_scales_use_all_supported_reference_people(self):
        rng = np.random.default_rng(43)
        eligible = rng.normal(size=(50, 8))
        extra = rng.normal(size=(12, 8)) + 3
        ref = np.vstack((eligible, extra))
        valid = rng.normal(size=(23, 8))
        train, val = m._pcs(eligible, valid, ref, 5)
        centered = ref - ref.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        load = vt[:5].T
        load *= np.sign(load[np.argmax(abs(load), axis=0), np.arange(5)])
        scales = np.std(centered @ load, axis=0)
        np.testing.assert_allclose(train, (eligible-ref.mean(0))@load / scales, atol=1e-11)
        np.testing.assert_allclose(val, (valid-ref.mean(0))@load / scales, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
