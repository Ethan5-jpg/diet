#!/usr/bin/env python3
"""Check V3 design package provenance/consistency; never reads study data."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
QA = ROOT / "outputs/researchwrite/hpp_multisystem_20261021/qa_logs"
VERSION = "multisystem_v3_20261023"
DOCS = [
    "HPP_CURRENT_PLAN.md",
    "HPP_Integrated_Multisystem_Plan_v3_2026-10-23.md",
    "HPP_Journal_Route_v3_2026-10-23.md",
    "HPP_Conference_Route_v3_2026-10-23.md",
    "HPP_Project_Roadmap_v3_2026-10-23.md",
    "HPP_Project_TODO_v3_2026-10-23.md",
    "HPP_Astra_Recommendations_Implementation_v3_2026-10-23.md",
]
EXPECTED_TASKS = [f"T{n:02}" for n in [0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,
                                       40,41,42,43,44,45,50,51,52,53,60,61,62,63,64,
                                       70,71,72,80,81,82,90,91]]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Structure(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.articles, self.external_assets = [], [], [], []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a: self.ids.append(a["id"])
        if tag == "article": self.articles.append(a.get("data-document"))
        if tag == "a" and "href" in a: self.links.append(a["href"])
        if tag in ["script", "link"] and (a.get("src") or a.get("href")):
            self.external_assets.append(a.get("src", a.get("href")))


def main():
    checks, failures, broken = {}, [], []
    def check(name, truth):
        checks[name] = bool(truth)
        if not truth: failures.append(name)
    for name in DOCS:
        p = REPORTS / name
        check(f"exists:{name}", p.exists())
        if not p.exists(): continue
        text = p.read_text()
        check(f"version:{name}", VERSION in text or name.startswith("HPP_Astra_Recommendations"))
        for target in re.findall(r"\]\(([^\n)]+)\)", text):
            if urlparse(target).scheme or target.startswith("#"): continue
            resolved = (p.parent / unquote(target.split("#")[0].strip("<>"))).resolve()
            if not resolved.exists(): broken.append({"source": name, "target": target})
    roadmap = (REPORTS / DOCS[4]).read_text()
    todo = (REPORTS / DOCS[5]).read_text()
    table_tasks = re.findall(r"^\| (T\d\d) `", roadmap, re.M)
    todo_tasks = re.findall(r"^- \[ \] (T\d\d)\b", todo, re.M)
    check("roadmap_exactly_38_unique_original_tasks", sorted(table_tasks) == sorted(EXPECTED_TASKS))
    check("todo_exactly_38_unique_original_tasks", sorted(todo_tasks) == sorted(EXPECTED_TASKS))
    check("no_scientific_todo_marked_complete", not re.search(r"^- \[[xX]\]", todo, re.M))
    registry = json.loads((REPORTS / "HPP_Cohort_Analysis_Register_v3_2026-10-23.json").read_text())
    check("registry_version", registry["version"] == VERSION)
    check("nine_modules", len(registry["modules"]) == 9)
    check("counts_not_recomputed", not registry["counts_recomputed"])
    modules = {m["analysis_id"]: m for m in registry["modules"]}
    for module_id, m in modules.items():
        check(f"null_counts:{module_id}", m["final_n"] is None and all(n is None for n in m["counts"].values()))
        check(f"null_q_rank:{module_id}", m["actual_q"] is None and m["actual_rank"] is None)
        for parent in m.get("candidate_parent_intersection", []):
            check(f"base_parent:{module_id}:{parent}", parent.startswith("E_"))
    for module_id in ["D_TotalFat", "D_VAT", "D_Liver", "D_BMI", "P_VAT_TotalFat"]:
        m = modules[module_id]
        check(f"no_unrelated_cgm_requirement:{module_id}", "cgm" not in m["required_sources"] and not m["requires_food_cgm_overlap"])
    check("Common4_not_gate", modules["Common4"]["gates_other_modules"] is False)
    daily = modules["Daily_CGM"]
    check("daily_primary_D_Glu", daily["primary_score_source"] == "D_Glu" and not daily["validates_pair_score_automatically"])
    check("daily_no_body_liver_labels", not daily["requires_body_or_liver_labels"])
    policy = registry["microbiome_time_policy"]
    check("no_fixed_lag", policy["fixed_max_lag_days"] is None)
    check("visit_identity_qc_verified", policy["visit_assignment_must_be_verified"] and policy["sample_identity_and_qc_must_pass"])
    for key in ["require_sampling_before_diet_or_outcome", "require_oral_gut_same_calendar_date", "require_unique_sampling_date_per_site_visit", "require_exact_sampling_date_for_main_eligibility", "default_choose_nearest_to_outcome"]:
        check(f"not_hard_filter:{key}", policy[key] is False)
    check("time_sensitivity_optional_not_gate", not policy["time_sensitivity"]["required"] and not policy["time_sensitivity"]["gates_main_analysis"])
    split = registry["split_policy"]
    check("roles_before_supervision", split["final_role_and_ratio_freeze_before_real_supervised_score_learning"] and not split["after_score_learning_role_reassignment"])
    check("all_before_first_confirmation", split["all_confirmatory_modules_freeze_before_first_confirmation"])
    check("all_scientific_gates_pending", not any(registry["acceptance_gates"].values()))
    h = registry["multiplicity"]
    ids = [x["hypothesis_id"] for x in h["hypotheses"]]
    check("fifteen_unique_hypotheses", len(ids) == len(set(ids)) == 15)
    check("Holm_default_005", h["method"] == "Holm" and h["alpha"] == .05)
    check("no_root_or_domain_gate", not h["Common4_root_gate"] and not h["independent_domain_p_gates_pair"])
    check("rank_failure_not_hypothesis_change", "no_generalized_inverse_or_posthoc" in h["rank_failure_policy"])
    ip = h["interval_policy"]
    check("default_descriptive_CI", not ip["formal_component_direction_threshold_equivalence_claims_enabled"] and not ip["optional_pre_unblinding_upgrade"]["enabled"])
    check("daily_estimand_separate", "within_person_fixed_effects_weighted_projection" in registry["estimand_contract"]["daily_operational_projection"])
    manifest = json.loads((REPORTS / "HPP_Project_Review_Manifest_v3_2026-10-23.json").read_text())
    html_file = REPORTS / manifest["output"]
    html = html_file.read_text()
    check("html_hash", sha(html_file) == manifest["html_sha256"])
    check("all_reader_sources_match", all(sha(REPORTS / d["filename"]) == d["sha256"] for d in manifest["documents"]))
    check("no_formula_placeholders", "HPPV3FORMULA" not in html)
    parser = Structure();parser.feed(html)
    check("seven_articles", len(parser.articles) == 7)
    check("unique_html_ids", len(parser.ids) == len(set(parser.ids)))
    check("offline_no_external_runtime", not parser.external_assets)
    for link in parser.links:
        if link.startswith("#"):
            if unquote(link[1:]) not in parser.ids: broken.append({"source": html_file.name, "target": link})
        elif not urlparse(link).scheme:
            if not (html_file.parent / unquote(link.split("#")[0])).exists():
                broken.append({"source": html_file.name, "target": link})
    check("no_broken_local_links", not broken)
    old = json.loads((QA / "v3_input_manifest_20261023.json").read_text())
    changed_history = [x["path"] for x in old["historical_reports"] if sha(ROOT / x["path"]) != x["sha256"]]
    check("historical_reports_unchanged", not changed_history)
    check("pre_revision_snapshots_intact", all(sha(ROOT / x["snapshot"]) == x["sha256"] for x in old["snapshots"]))
    check("independent_implementation_QA_exists", (QA / "v3_implementation_independent_review_20261023.md").exists())
    state = json.loads((ROOT / "outputs/researchwrite/hpp_multisystem_20261021/state.json").read_text())
    check("state_current_version_and_implementation", state["current_version"] == VERSION and state["latest_review_revisions_applied"])
    check("state_no_new_astra_rereview_claim", state["current_version_astra_full_review_run"] is False)
    check("state_scientific_gates_pending", not state["protocol_inferentially_frozen"] and not state["new_scientific_models_run"] and not state["new_simulations_run"])
    package_files = [REPORTS / n for n in DOCS] + [REPORTS / "HPP_Cohort_Analysis_Register_v3_2026-10-23.json", html_file,
        REPORTS / "build_project_register_v3_20261023.py", REPORTS / "build_project_review_v3_20261023.mjs", Path(__file__),
        ROOT / "outputs/researchwrite/hpp_multisystem_20261021/state.json",
        ROOT / "outputs/researchwrite/hpp_multisystem_20261021/MEMO_v3_implementation_20261023.md"]
    result = {"version": VERSION, "date": "2026-10-23", "validation_scope": "design_document_configuration_provenance_and_reader_structure_only",
              "passed": not failures, "check_count": len(checks), "checks": checks,
              "failures": failures, "broken_local_links": broken, "changed_historical_reports": changed_history,
              "roadmap_task_count": len(table_tasks), "todo_task_count": len(todo_tasks),
              "module_count": len(modules), "hypothesis_count": len(ids), "html_article_count": len(parser.articles),
              "python": sys.executable, "python_version": sys.version.split()[0],
              "browser_visual_inspection": "not_run_prior_file_protocol_policy_respected_no_workaround",
              "server_connected": False, "data_qc_run": False, "new_models_run": False, "new_simulations_run": False,
              "scientific_acceptance_verified_by_document_QA": False,
              "artifacts": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in package_files]}
    target = QA / "v3_implementation_validation_20261023.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "failures": failures, "broken_links": broken,
                      "tasks": len(table_tasks), "modules": len(modules), "hypotheses": len(ids)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
