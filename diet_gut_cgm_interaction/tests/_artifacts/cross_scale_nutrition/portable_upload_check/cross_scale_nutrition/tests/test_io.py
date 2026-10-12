import json
from pathlib import Path
import shutil

import pytest

from crossscale.common import sha256
from crossscale.inputs import open_reference, reproduce_reference, REFERENCE_FILES
from crossscale.reporting import saved_screens

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests/_artifacts/daily_within_person/alcohol_revision_20261011/release_result"


@pytest.fixture
def copied_reference(tmp_path):
    if not REFERENCE.is_dir():
        pytest.skip("Optional historical synthetic artifact is absent from upload")
    dest = tmp_path / "reference"
    dest.mkdir()
    for name in ["manifest.json"] + REFERENCE_FILES:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REFERENCE / name, path)
    return dest


def test_historical_reproduction_and_readonly_reference(copied_reference, protocol):
    before = {str(p): sha256(p) for p in copied_reference.rglob("*") if p.is_file()}
    panel, config, results, _, _, verify = open_reference(copied_reference, protocol, True)
    checks = reproduce_reference(panel, config, results, protocol, True)
    assert len(checks) == 8 and checks.passed.all()
    verify()
    assert before == {str(p): sha256(p) for p in copied_reference.rglob("*") if p.is_file()}


@pytest.mark.parametrize("file", ["private/daily_panel.csv", "tables/10_interactions.csv", "analysis_config.json", "protocol.json"])
def test_reference_hash_tampering_refused(copied_reference, protocol, file):
    p = copied_reference / file
    p.write_text(p.read_text() + "\n")
    with pytest.raises(ValueError, match="Saved reference changed"):
        open_reference(copied_reference, protocol, True)


def test_real_synthetic_mismatch_refused(copied_reference, protocol):
    with pytest.raises(ValueError, match="mode mismatch"):
        open_reference(copied_reference, protocol, False)


def test_runtime_and_config_changes_refused(copied_reference, protocol):
    m = json.loads((copied_reference / "manifest.json").read_text())
    m["code_sha256"]["run_revision.py"] = "modified"
    (copied_reference / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="runtime fingerprint"):
        open_reference(copied_reference, protocol, True)


def test_saved_screens_detect_corrupted_table(tmp_path):
    (tmp_path / "tables").mkdir()
    names = ["SCREEN_1.txt", "SCREEN_2.txt", "SCREEN_3.txt", "tables/10_interactions.csv",
             "tables/11_comparisons.csv", "tables/12_model_diagnostics.csv", "config.json"]
    for name in names:
        (tmp_path / name).write_text("SYNTHETIC\n")
    m = dict(status="COMPLETE", output_sha256={name: sha256(tmp_path / name) for name in names})
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert len(saved_screens(tmp_path)) == 3
    (tmp_path / "tables/11_comparisons.csv").write_text("changed")
    with pytest.raises(ValueError, match="Saved result changed"):
        saved_screens(tmp_path)
