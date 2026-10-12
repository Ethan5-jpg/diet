import numpy as np
import pandas as pd
import pytest

from daily.common import scoped
from daily.inputs import species_mapping, timing_table


def test_taxon_token_exact_and_ambiguity(config):
    cols = ["k__B|s__Eubacterium_rectale", "s__Ruthenibacterium_lactatiformans", "s__Eubacterium_rectale_extra"]
    assert species_mapping(cols, config["taxa"])[0]["column"] == cols[0]
    with pytest.raises(ValueError, match="expected one exact"):
        species_mapping(cols + ["Eubacterium_rectale"], config["taxa"])


def test_visit_and_duplicate_identity_are_not_guessed(config):
    d = pd.DataFrame(dict(participant_id=["001", "002"], cohort="10k"))
    with pytest.raises(ValueError, match="identity"):
        scoped(d, config, "microbiome", [])
    d["research_stage"] = "00_00_visit"
    d.loc[1, "participant_id"] = "001.0"
    with pytest.raises(ValueError, match="duplicate identity"):
        scoped(d, config, "microbiome", [])


def test_ambiguous_sampling_date_remains_unknown(config, tmp_path):
    path = tmp_path / "dates.csv"
    pd.DataFrame(dict(participant_id=["a", "a"], cohort="10k", research_stage="00_00_visit",
                      collection_date=["2024-01-01", "2024-01-02"], sample_name=["x", "y"])).to_csv(path, index=False)
    m = pd.DataFrame(dict(participant_id=["a"]))
    out = timing_table(m, path, config)
    assert out.loc[0, "sample_date"] == ""
    assert out.loc[0, "timing_status"].startswith("UNKNOWN_AMBIGUOUS")
    m["sample_name"] = "y"
    out = timing_table(m, path, config)
    assert out.loc[0, "sample_date"] == "2024-01-02"
    assert out.loc[0, "timing_status"] == "MATCHED_SAMPLE"


@pytest.mark.parametrize("empty", [False, True])
def test_optional_dates_without_target_visit_remain_unknown(config, tmp_path, empty):
    path = tmp_path / "dates.csv"
    d = pd.DataFrame(dict(participant_id=["a"], cohort=["10k"], research_stage=["other_visit"],
                          collection_date=["2024-01-01"]))
    d.iloc[:0 if empty else 1].to_csv(path, index=False)
    out = timing_table(pd.DataFrame(dict(participant_id=["a"])), path, config)
    assert out.loc[0, "sample_date"] == ""
    assert out.loc[0, "timing_status"] == "UNKNOWN_NO_MATCHING_VISIT"
    # Required participant sources must still reject a wrong or empty visit.
    with pytest.raises(ValueError, match="no rows"):
        scoped(d.iloc[:0 if empty else 1], config, "microbiome", [])
