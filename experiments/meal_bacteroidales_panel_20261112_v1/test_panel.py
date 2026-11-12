"""Verify strict taxa mapping, panel scaling, and matched loss accounting."""

import unittest

import numpy as np

from panel_design import SPECIES, panel_mass, score_losses, species_columns


class PanelTest(unittest.TestCase):
    def test_exact_terminal_species_only(self):
        taxa = ["g__Alistipes|" + SPECIES[1], SPECIES[2], SPECIES[0], SPECIES[0] + "_related"]
        self.assertEqual(species_columns(taxa), [2, 0, 1])
        with self.assertRaisesRegex(ValueError, "align exactly once"):
            species_columns(taxa + [SPECIES[0]])

    def test_fixed_panel_mass_and_detection(self):
        raw = np.array([[.1, .2, 0., .7], [0., 0., 0., 1.]])
        score, seen = panel_mass(raw, [*SPECIES, "s__Other"])
        np.testing.assert_allclose(score, np.log1p(np.array([.3, 0.]) / 1e-4))
        np.testing.assert_array_equal(seen, [1, 1, 0])
        with self.assertRaisesRegex(ValueError, "abundance scale"):
            panel_mass(raw * 2, [*SPECIES, "s__Other"])

    def test_joint_gain_uses_the_pc_referent(self):
        losses = np.array([[10., 20.], [8., 16.], [7., 14.], [6., 12.], [10., 20.]])
        out = score_losses(losses, 1)
        self.assertAlmostEqual(out["panel_vs_baseline"]["delta_r2"], .2)
        self.assertAlmostEqual(out["joint_vs_pcs"]["delta_r2"], .1)
        self.assertAlmostEqual(out["panel_vs_pcs"]["delta_r2"], -.1)
        np.testing.assert_allclose(out["joint_vs_pcs"]["interval"], [.1, .1])
