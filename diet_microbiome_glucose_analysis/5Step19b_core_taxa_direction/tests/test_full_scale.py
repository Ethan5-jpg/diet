"""Full-size SYNTHETIC fixtures; the reported study effects are never fabricated here."""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from test_direction_synthesis import fixture, m


def full_fixture(root):
    source, models, paths, _ = fixture(root)
    dictionary = pd.read_csv(m.PACKAGE / "config/direction_dictionary.csv")
    exposures = dictionary.loc[dictionary.kind.eq("exposure"), "field"].tolist()
    outcomes = {domain: dictionary.loc[dictionary.domain.eq(domain), "field"].tolist() for domain in m.DOMAINS}
    taxa = ["k__Synthetic|g__Test{:03d}|s__Synthetic_{:03d}".format(i, i) for i in range(109)]
    allowed = {}
    for i, taxon in enumerate(taxa):
        if i < 4:
            allowed[taxon] = m.DOMAINS
        elif i < 21:
            allowed[taxon] = m.DOMAINS[:2]
        elif i < 39:
            allowed[taxon] = [m.DOMAINS[0]]
        elif i < 107:
            allowed[taxon] = [m.DOMAINS[1]]
        else:
            allowed[taxon] = [m.DOMAINS[2]]
    rows = []
    seen = set()
    # Round robin across taxa, then exposure/outcome: enough unique paths and all taxa represented.
    for domain in m.DOMAINS:
        count = 0
        target = m.EXPECTED[domain]
        for exposure in exposures:
            for outcome in outcomes[domain]:
                for taxon in taxa:
                    if domain in allowed[taxon] and count < target:
                        row = (exposure, outcome, taxon)
                        rows.append(row + (domain,))
                        seen.add(row)
                        count += 1
        assert count == target
    assert len(rows) == 871
    # Add non-highest candidates without introducing more taxa into the frozen 871.
    for exposure in exposures:
        for domain in m.DOMAINS:
            for outcome in outcomes[domain]:
                for taxon in taxa:
                    row = (exposure, outcome, taxon)
                    if row not in seen and len(rows) < 1490:
                        rows.append(row + (domain,))
                        seen.add(row)
    data = pd.DataFrame(rows, columns=m.KEYS + ["phenotype_domain"])
    n = len(data)
    index = np.arange(n)
    highest = index < 871
    masks = {"primary": index < 1003,
             "strict": (index < 887) | ((index >= 1003) & (index < 1030)),
             "protocol": (index < 871) | ((index >= 887) & (index < 956)) | ((index >= 1030) & (index < 1033))}
    frozen = data[m.KEYS].copy()
    frozen["highest_robustness"] = highest
    for tag, path in models.items():
        template = pd.read_csv(path)
        d = template.iloc[np.zeros(n, dtype=int)].reset_index(drop=True).copy()
        for key in m.KEYS:
            d[key] = data[key]
        a = np.where(index % 2 == 0, .15, -.15)
        b = np.where(index % 3 == 0, .21, -.21)
        # Small true between-model changes so cross-run comparisons exercise real numbers.
        factor = {"primary": 1, "strict": .9, "protocol": .97}[tag]
        d["a_diet_to_microbiome"] = a * factor
        d["b_microbiome_to_cgm"] = b
        d["indirect_effect"] = a * factor * b
        d["direct_effect"] = d["indirect_effect"]
        d["total_effect"] = 2 * d["indirect_effect"]
        for effect in ["indirect", "direct", "total"]:
            width = np.where(masks[tag], .01, .2)
            d[effect + "_CI95_lower"] = d[effect + "_effect"] - width
            d[effect + "_CI95_upper"] = d[effect + "_effect"] + width
        d["FDR_BH_within_score_outcome"] = np.where(masks[tag], .01, .9)
        d["bootstrap_p_indirect"] = np.where(masks[tag], .001, .8)
        d["mediation_FDR05_primary"] = masks[tag]
        d["bridge_direction_consistent_mediator"] = masks[tag]
        d.to_csv(path, index=False)
        for base, col in [("indirect_effect", "indirect_effect"), ("N", "N"),
                          ("FDR", "FDR_BH_within_score_outcome"), ("bootstrap_p", "bootstrap_p_indirect")]:
            frozen[base + "_" + tag] = d[col]
        for base in ["significant", "consistent", "robust"]:
            frozen[base + "_" + tag] = masks[tag]
    frozen["primary_retained_strict"] = masks["primary"] & masks["strict"]
    frozen["primary_retained_protocol"] = masks["primary"] & masks["protocol"]
    annotated = frozen.copy()
    annotated["phenotype_domain"] = data.phenotype_domain
    h = annotated[highest].copy()
    summary = h.groupby("species").agg(phenotype_domains=("phenotype_domain", "nunique"),
                                      highest_robust_paths=("species", "size"),
                                      diet_exposures=("exposure", "nunique"),
                                      cgm_outcomes=("outcome_field", "nunique")).reset_index()
    tables = {"18f_all": frozen, "18f_high": frozen[highest], "18g": annotated, "19a": h,
              "19a_species": summary, "19a_pan": summary[summary.phenotype_domains.eq(3)],
              "19a_cross": summary[summary.phenotype_domains.eq(2)]}
    for key, frame in tables.items():
        frame.to_csv(paths[key], index=False)
    return source


class FullScaleTests(unittest.TestCase):
    def test_full_default_counts_and_outputs(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = full_fixture(root)
            args = m.parser().parse_args(["--source-root", str(source), "--output-parent", str(root / "outputs")])
            args.synthetic = True
            # Plot verification is available when optional matplotlib is installed.
            args.no_plots = importlib.util.find_spec("matplotlib") is None
            with contextlib.redirect_stdout(io.StringIO()):
                status, out = m.run(args)
            self.assertEqual(status, 0, (out / "19b_SCREENSHOT_SUMMARY.txt").read_text())
            result = pd.read_csv(out / "19b_path_direction_details.csv")
            self.assertEqual(len(result), 871)
            self.assertEqual(result.species.nunique(), 109)
            self.assertEqual(result.domain_breadth.value_counts().index.sort_values().tolist(), [1, 2, 3])
            self.assertEqual(len(list(out.glob("19b_CORE_*_SCREENSHOT.txt"))), 4)
            self.assertEqual(len(pd.read_csv(out / "19b_representative_paths.csv")), 12)
            if not args.no_plots:
                self.assertEqual(len(list(out.glob("*.png"))), 23)
                self.assertTrue(all(p.stat().st_size > 1000 for p in out.glob("*.png")))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["data_kind"], "SYNTHETIC_TEST_ONLY")
            self.assertTrue(manifest["all_inputs_unchanged_verified"])
            if os.environ.get("HPP_STEP19B_KEEP_QA"):
                destination = Path(os.environ["HPP_STEP19B_KEEP_QA"])
                # Deliberately separate synthetic previews from the production outputs folder.
                destination.mkdir(parents=True, exist_ok=True)
                for file in out.iterdir():
                    if file.suffix == ".png" or "SCREENSHOT" in file.name or file.name == "manifest.json":
                        shutil.copy2(file, destination / file.name)


if __name__ == "__main__":
    unittest.main()
