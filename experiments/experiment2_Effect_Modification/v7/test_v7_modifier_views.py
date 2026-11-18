"""Numerical and training-role checks for the expanded gut representations."""

import unittest

import numpy as np
from scipy.linalg import helmert

import v7_modifier_views as views


def sparse_profiles(seed=931):
    rng = np.random.default_rng(seed)
    mass = rng.lognormal(size=(100, 20))
    mass[:, 0] += 1.0
    mass[:, 1:18] *= rng.uniform(size=(100, 17)) < .55
    mass[:95, 18:] = 0  # Rare in training, positive throughout validation.
    return mass / mass.sum(axis=1, keepdims=True)


class ModifierViewTests(unittest.TestCase):
    def test_fixed_fourier_maps_are_centered_and_isolated(self):
        rng = np.random.default_rng(41)
        for band, (width, factor) in views.RFF_BANDS.items():
            train = rng.normal(size=(60, width))
            valid = rng.normal(size=(12, width)) + 2
            mapped_train, mapped_valid = views.fit_rff(train, valid, band)
            # Complex exponentials provide an independent representation of the map.
            directions = np.random.default_rng(views.RFF_SEED + width).normal(
                size=(width, views.N_FREQUENCIES))
            def complex_map(x):
                phase = (x - train.mean(axis=0)) @ directions * np.sqrt(2 * factor / width)
                z = np.exp(1j * phase) / np.sqrt(views.N_FREQUENCIES)
                return np.column_stack((z.real, z.imag))
            self.assertEqual(mapped_train.shape, (60, 24))
            offset = complex_map(train).mean(axis=0)
            np.testing.assert_allclose(mapped_train, complex_map(train) - offset, atol=1e-13)
            np.testing.assert_allclose(mapped_valid, complex_map(valid) - offset, atol=1e-13)
            valid[0] += 200
            train_again, valid_again = views.fit_rff(train, valid, band)
            np.testing.assert_array_equal(train_again, mapped_train)
            np.testing.assert_array_equal(valid_again[1:], mapped_valid[1:])

    def test_taxon_selection_and_three_numeric_views(self):
        profiles = sparse_profiles()
        train, valid = profiles[:95], profiles[95:]
        outputs, selected, prevalence = views.fit_composition(train, valid)
        log = np.log(np.clip(profiles, views.LOG_FLOOR, None))
        clr = log - log.mean(axis=1, keepdims=True)
        ranked = sorted([j for j in range(train.shape[1]) if np.mean(train[:, j] > 0) >= .1],
                        key=lambda j: (-clr[:95, j].std(), j))
        expected = ranked[:12]
        np.testing.assert_array_equal(selected, expected)
        np.testing.assert_allclose(prevalence, (train[:, expected] > 0).mean(axis=0))
        # Scipy's forward Helmert contrasts, reversed, give the correct pivot basis.
        pivots = -log[:, expected[::-1]] @ helmert(len(selected)).T
        pivots = pivots[:, ::-1]
        detect = (profiles[:, expected] > 0)
        intensities = np.zeros_like(detect, dtype=float)
        for k, j in enumerate(expected):
            positive_mean = log[:95, j][train[:, j] > 0].mean()
            intensities[detect[:, k], k] = log[detect[:, k], j] - positive_mean
        for key, values in (("clr", clr[:, expected]), ("pivot", pivots),
                            ("detect_intensity", np.column_stack((detect, intensities)))):
            centered = values - values[:95].mean(axis=0)
            for got, want in zip(outputs[key], (centered[:95], centered[95:])):
                np.testing.assert_allclose(got, want, atol=2e-13)

        changed = valid.copy()
        changed[0] = np.eye(1, train.shape[1], train.shape[1] - 1)[0]
        again, selected_again, _ = views.fit_composition(train, changed)
        np.testing.assert_array_equal(selected, selected_again)
        for key in outputs:
            np.testing.assert_array_equal(outputs[key][0], again[key][0])
            np.testing.assert_array_equal(outputs[key][1][1:], again[key][1][1:])

    def test_no_eligible_taxa_and_variable_widths(self):
        train = np.eye(30)
        valid = np.full((4, 30), 1 / 30)
        zero_views, selected, _ = views.fit_composition(train, valid)
        self.assertEqual(len(selected), 0)
        for name, pair in zero_views.items():
            for n, values in zip((30, 4), pair):
                self.assertEqual(values.shape, (n, views.COMPOSITION_WIDTHS[name]))
                np.testing.assert_array_equal(values, np.zeros_like(values))
        # Tied all-observed columns have deterministic ordering; the one
        # unobserved training taxon never acquires a validation-only coordinate.
        small = np.full((8, 4), .25)
        small[:, 0] += .25
        small[:, 3] = 0
        result, selected, _ = views.fit_composition(small, small[:2])
        self.assertEqual(set(selected), {0, 1, 2})
        self.assertLess(list(selected).index(1), list(selected).index(2))
        for pair in result.values():
            for values in pair:
                np.testing.assert_allclose(values, 0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "FRACTIONAL"):
            views.fit_composition(small * 100, small[:2])


if __name__ == "__main__":
    unittest.main()
