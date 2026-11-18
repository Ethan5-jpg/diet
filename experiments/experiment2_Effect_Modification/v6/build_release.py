"""Local maintainer utility: derive the route inventory and upload checksums.

Do not regenerate checksums on the server to bypass a failed upload check.
Run tests after changes, then regenerate here before uploading the whole V6.
"""
import csv
import hashlib
import json
from pathlib import Path

from experiment import grid, route_names, settings
from run_v6 import PINS, VERSION

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    original = {"person_inputs.py": "v2/person_inputs.py", "moderation_models.py": "v2/moderation_models.py",
                "fermentation_score.py": "v3/fermentation_score.py"}
    provenance = {}
    for name, source in original.items():
        vendor, old = ROOT / "adapters" / name, ROOT.parent / source
        if sha(vendor) != sha(old):
            raise RuntimeError("VENDORED_ADAPTER_CHANGED " + name)
        provenance["adapters/" + name] = dict(original="experiment2_Effect_Modification/" + source,
                                             sha256=sha(vendor), exact_copy=True)
    write(ROOT / "adapter_provenance.json", provenance)
    write(ROOT / "config.json", dict(version=VERSION, environment=PINS,
        note="Derived documentation; runtime grid is fixed in experiment.py and interaction_candidates.py",
        discovery_only=True, people=3343, households=3059, outcome="D_Glu", exposure="fiber_density",
        contrast=[10, 15], bootstrap_draws=1999, outer_repeats=2, outer_folds=5, inner_folds=3,
        single_families=len(grid()), nonzero_settings=len(settings()), reported_routes=len(route_names()),
        primary="nested_selector", ratio="100 * delta_r2 / full_r2", routes=route_names()))
    with (ROOT / "routes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["route", "setting", "site", "view", "n_components", "kind", "strength", "n_dose", "latent_rank", "l1_ratio", "rff_features", "seed"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for name, candidate in settings():
            from experiment import setting_key
            writer.writerow(dict(route=name, setting=setting_key(name, candidate), **candidate.to_dict()))
    (ROOT / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k, v in PINS.items()) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if (not path.is_file() or any(p in ("runs", "__pycache__") for p in rel.parts)
                or path.name == "upload_manifest.json" or path.suffix not in
                (".py", ".json", ".csv", ".md", ".sh", ".txt")):
            continue
        files[str(rel)] = sha(path)
    write(ROOT / "upload_manifest.json", dict(version=VERSION, files=files,
        scope="Code, tests, methods and historical aggregate pairs only; no native individual HPP records"))
    print(json.dumps(dict(files=len(files), families=len(grid()), settings=len(settings()), routes=len(route_names()))))


if __name__ == "__main__":
    main()
