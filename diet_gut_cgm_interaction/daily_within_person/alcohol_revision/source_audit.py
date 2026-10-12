"""Same rules as the 18:09 screenshot audit; unknown alcohol is never filled."""
import pandas as pd
from daily.common import read_csv, scoped, date_only, number, require

RENAME = {
    "alcohol_g_total": "alcohol_amed_g", "event_count": "amed_events",
    "excluded_missing_labels_event_count": "excluded", "alcoholic_drink_event_count": "drinks",
    "alcohol_observed_event_count": "obs", "alcohol_energy_formula_event_count": "formula",
    "alcohol_standard_abv_event_count": "abv", "alcohol_zero_fallback_event_count": "zero",
    "alcohol_unresolved_event_count": "unresolved", "alcohol_invalid_observed_event_count": "invalid",
}


def attach_alcohol(panel, source, config):
    required = ["participant_id", "cohort", "research_stage", "collection_date"] + list(RENAME)
    require(not set(RENAME.values()).intersection(panel.columns), "Panel already contains revised alcohol columns")
    a = scoped(read_csv(source, required, usecols=required), config, "AMED", [],
               unique_person=False, allow_empty=True)
    scoped_rows = len(a)
    a["date"] = date_only(a.collection_date, config["time"]["local_timezone"])
    bad_dates = int(a.date.isna().sum())
    a = a.loc[a.date.notna()].rename(columns=RENAME)
    keys = ["participant_id", "date"]
    duplicates = a.duplicated(keys, keep=False)
    duplicate_keys = a.loc[duplicates, keys].drop_duplicates().assign(ambiguous=True)
    a = a.loc[~duplicates, keys + list(RENAME.values())].copy()
    for field in RENAME.values():
        a[field] = number(a[field])
    p = panel.merge(duplicate_keys, on=keys, how="left", validate="one_to_one")
    p = p.merge(a, on=keys, how="left", validate="one_to_one", indicator=True)
    joined = p["_merge"].eq("both")
    counts = [v for v in RENAME.values() if v != "alcohol_amed_g"]
    valid = p[counts].notna().all(axis=1) & p[counts].ge(0).all(axis=1) & p[counts].mod(1).eq(0).all(axis=1)
    closure = p[["obs", "formula", "abv", "zero", "unresolved"]].sum(axis=1, min_count=5).eq(p.drinks)
    closure &= p.drinks.le(p.amed_events) & (p.drinks.gt(0) | p.alcohol_amed_g.eq(0))
    event_match = (p.amed_events + p.excluded).eq(p.events)
    safe = joined & valid & closure & event_match & p.alcohol_amed_g.ge(0)
    safe &= p[["excluded", "zero", "unresolved", "invalid"]].eq(0).all(axis=1)
    p["alcohol_recovered_ok"] = safe
    p["alcohol_observed_ok"] = safe & p.formula.eq(0) & p.abv.eq(0)
    qc = p.primary_eligible & p.diet_pass & p.cgm_pass
    audit = dict(scoped_amed_rows=scoped_rows, invalid_date_rows=bad_dates,
                 qc_days=int(qc.sum()), unique_matched_days=int((qc & joined).sum()),
                 ambiguous_days=int((qc & p.ambiguous.eq(True)).sum()),
                 unmatched_days=int((qc & ~joined & ~p.ambiguous.eq(True)).sum()),
                 event_count_mismatch_days=int((qc & joined & ~event_match).sum()),
                 invalid_source_counts_days=int((qc & joined & ~valid).sum()),
                 source_closure_failure_days=int((qc & joined & ~closure).sum()),
                 candidate_observed_days=int((qc & p.alcohol_observed_ok).sum()),
                 candidate_recovered_days=int((qc & safe).sum()))
    for field in ["drinks", "obs", "formula", "abv", "zero", "unresolved", "invalid", "excluded"]:
        audit["flag_days_" + field] = int((qc & joined & p[field].gt(0)).sum())
    return p, audit


def sample_digest(d):
    import hashlib
    return hashlib.sha256("\n".join(sorted(d.day_id)).encode()).hexdigest()


def checkpoints(panel, config):
    from revision_models import choose_sample, EXPOSURES
    rows = []
    policies = [("no_alcohol_requirement", "all_days_no_alcohol_control"),
                ("audit_observed_only", "primary"), ("audit_recovered", "alcohol_recovered")]
    for exposure in EXPOSURES:
        for taxon in [t["id"] for t in config["taxa"]]:
            for label, spec in policies:
                d, flow = choose_sample(panel, config, exposure, taxon, "SD", spec)
                complete = next(row for row in flow if row["stage"] == "complete_model_fields")
                rows.append(dict(exposure=exposure, taxon=taxon, source=label,
                                 complete_people=int(complete["n_people"]), complete_days=int(complete["n_days"]),
                                 final_people=int(d.participant_id.nunique()), final_days=len(d)))
    return rows


def verify_paired_samples(panel, config):
    from revision_models import choose_sample, EXPOSURES, OUTCOMES
    rows = []
    for adjusted, unadjusted in [("primary", "primary_no_alcohol_control"),
                                ("alcohol_recovered", "recovered_no_alcohol_control")]:
        for exposure in EXPOSURES:
            for taxon in [t["id"] for t in config["taxa"]]:
                for outcome in OUTCOMES:
                    a, _ = choose_sample(panel, config, exposure, taxon, outcome, adjusted)
                    b, _ = choose_sample(panel, config, exposure, taxon, outcome, unadjusted)
                    require(sample_digest(a) == sample_digest(b), "Adjusted/unadjusted person-day sample differs")
                    rows.append(dict(adjusted=adjusted, unadjusted=unadjusted, exposure=exposure,
                                     taxon=taxon, outcome=outcome, n_people=a.participant_id.nunique(),
                                     n_days=len(a), sample_sha256=sample_digest(a), identical=True))
    return pd.DataFrame(rows)
