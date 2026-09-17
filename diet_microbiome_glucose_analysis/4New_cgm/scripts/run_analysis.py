#!/usr/bin/env python3
"""Run an isolated four-CGM association analysis using the legacy OLS process."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import uuid

import numpy as np
import pandas as pd
import scipy

import new_cgm_data as data
import new_cgm_models as models

VERSION = "2026-09-15.2"


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def write_csv(frame, path):
    frame.to_csv(str(path), index=False, encoding="utf-8")


def model_summary(result, branch):
    rows = []
    for key, group in result.groupby(["analysis_set", "model", "outcome"], sort=False):
        computed = group.status.eq("computed")
        rows.append({"branch": branch, "analysis_set": key[0], "model": key[1], "outcome": key[2],
                     "tests_planned": len(group), "computed": int(computed.sum()),
                     "failed": int((~computed).sum()), "N_min": int(group.N.min()), "N_max": int(group.N.max()),
                     "FDR_family_lt05": int(group.significant_family_05.sum()),
                     "FDR_global_lt05": int(group.significant_global_05.sum())})
    return pd.DataFrame(rows)


def summary_markdown(summary, diet_results, micro_results, run_id):
    lines = ["# 新四指标关联分析运行摘要", "", "运行 ID：`%s`。" % run_id,
             "", "采用原流程 OLS 与常规标准误；结果是关联，不作因果解释。", "",
             "## 模型完成情况", "", "```text", summary.to_string(index=False), "```", "",
             "## 主人群 Model 2 饮食结果", "", "```text",
             diet_results.loc[diet_results.analysis_set.eq("primary") & diet_results.model.eq(2),
                              ["exposure", "outcome", "N", "beta", "CI95_lower", "CI95_upper", "p_value", "FDR_family", "FDR_global", "status"]].to_string(index=False),
             "```", "", "## 主人群 Model 2 菌群：每个结局按 p 排列的前十项", ""]
    for outcome in data.OUTCOMES:
        frame = micro_results.loc[micro_results.analysis_set.eq("primary") & micro_results.model.eq(2) & micro_results.outcome.eq(outcome)]
        lines += ["### " + outcome, "", "```text", frame.sort_values("p_value").head(10)[
            ["species", "N", "beta", "p_value", "FDR_family", "FDR_global", "status"]].to_string(index=False), "```", ""]
    lines += ["## 解释范围", "", "- 不根据旧分析显著性筛选本次暴露、结局或物种。",
              "- TAR180 零值集中、比例偏态及合法高值仍保留；本轮复用旧模型，不表示已完成模型适用性或原始曲线核查。",
              "- TIR 为辅助结局，计入本轮预设四结局校正。",
              "- 失败模型在状态列记录，实际 p/q 留空；BH 内部用 p=1 保留计划检验规模。",
              "- 未运行饮食→菌群重算、路径筛选或中介分析。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(data.PROJECT_ROOT / "config/analysis.json"))
    parser.add_argument("--data-root", help="HPP/Data root, normally inferred from the uploaded folder")
    parser.add_argument("--check-only", action="store_true", help="Check inputs and save cohorts without fitting models")
    args = parser.parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    output = data.PROJECT_ROOT / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name in ("data", "models", "reports", "logs"):
        (output / name).mkdir()
    log_path = output / "logs/run.log"
    manifest = {"version": VERSION, "run_id": run_id, "status": "running", "check_only": args.check_only,
                "output_dir": str(output), "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__,
                "outcomes": data.OUTCOMES, "exposures": data.EXPOSURES,
                "estimation": "legacy_OLS_conventional_standard_errors",
                "old_association_results_used": False, "new_outcome_z_recomputed": False}
    code = 1
    with log_path.open("x", encoding="utf-8") as log:
        tee = Tee(sys.stdout, log)
        with redirect_stdout(tee), redirect_stderr(tee):
            try:
                print("=== 4New_cgm %s ===" % VERSION, flush=True)
                print("OUTPUT_DIR=" + str(output))
                # Check plotting dependencies before spending time fitting models.
                import new_cgm_visuals as visuals
                manifest["matplotlib"] = visuals.matplotlib.__version__
                config = data.load_config(args.config, args.data_root)
                manifest["config"] = config
                manifest["config_fingerprint"] = data.fingerprint(args.config)
                manifest["code_files"] = [data.fingerprint(p) for p in sorted((data.PROJECT_ROOT / "scripts").glob("*.py"))]
                print(json.dumps(config, ensure_ascii=False, indent=2))
                print("Checking five source tables and the four finalized outcomes...", flush=True)
                diet, microbiome, species, cgm_audit, overlap, fingerprints = data.prepare_tables(config)
                manifest["input_files"] = fingerprints
                manifest["cohort_summary"] = overlap
                print("\n=== INPUT / COHORT COUNTS ===")
                print(json.dumps(overlap, ensure_ascii=False, indent=2))
                print("\n=== CGM VALUES / Z PRESERVED ===")
                print(cgm_audit.to_string(index=False))
                write_csv(cgm_audit, output / "reports/cgm_input_audit.csv")
                write_csv(pd.DataFrame([overlap]), output / "reports/cohort_summary.csv")
                write_csv(diet, output / "data/diet_cgm_cohort.csv")
                write_csv(microbiome, output / "data/microbiome_cgm_cohort.csv")
                coverage = []
                for label, frame in (("diet", diet), ("microbiome", microbiome)):
                    fields = list(data.OUTCOMES.values()) + data.BASE_COVARIATES + ["bmi", "exclude_diabetes_or_a10"]
                    if label == "diet":
                        fields += [x[0] for x in data.EXPOSURES.values()] + ["mean_daily_energy_kcal", "alcohol_intake_g_day"] + data.FLAGS
                    for column in fields:
                        coverage.append({"branch": label, "column": column, "rows": len(frame),
                                         "available_n": int(models.complete_mask(frame, [column]).sum()),
                                         "missing_n": int((~models.complete_mask(frame, [column])).sum())})
                coverage = pd.DataFrame(coverage)
                write_csv(coverage, output / "reports/variable_coverage.csv")
                print("\n=== VARIABLE AVAILABILITY ===")
                print(coverage.to_string(index=False))
                if not args.check_only:
                    print("\n=== DIET -> NEW CGM: seven exposures, four outcomes ===", flush=True)
                    diet_result, diet_coef, diet_params, diet_members = models.fit_diet(diet, config)
                    write_csv(diet_result, output / "models/diet_cgm_all_models.csv")
                    write_csv(diet_coef, output / "reports/diet_all_coefficients.csv")
                    write_csv(diet_params, output / "reports/diet_design_parameters.csv")
                    write_csv(diet_members, output / "data/diet_model_membership.csv")
                    print("\n=== MICROBIOME -> NEW CGM ===", flush=True)
                    micro_result, micro_params, micro_members = models.fit_microbiome(microbiome, species, config)
                    write_csv(micro_result, output / "models/microbiome_cgm_all_models.csv")
                    write_csv(micro_params, output / "reports/microbiome_design_parameters.csv")
                    write_csv(micro_members, output / "data/microbiome_model_membership.csv")
                    write_csv(models.comparisons(diet_result, "diet"), output / "reports/diet_same_cohort_comparisons.csv")
                    write_csv(models.comparisons(micro_result, "microbiome"), output / "reports/microbiome_same_cohort_comparisons.csv")
                    summary = pd.concat([model_summary(diet_result, "diet"), model_summary(micro_result, "microbiome")], ignore_index=True)
                    write_csv(summary, output / "reports/model_summary.csv")
                    failures = pd.concat([diet_result.loc[diet_result.status.ne("computed")].assign(branch="diet"),
                                          micro_result.loc[micro_result.status.ne("computed")].assign(branch="microbiome")], ignore_index=True)
                    write_csv(failures, output / "reports/model_failures.csv")
                    manifest["failed_model_tests"] = len(failures)
                    manifest["status"] = "completed" if failures.empty else "completed_with_model_failures"
                    print("\n=== MODEL SUMMARY / FDR ===")
                    print(summary.to_string(index=False))
                    if not failures.empty:
                        print("\n=== MODEL FAILURES (first 20; see model_failures.csv) ===")
                        print(failures[["branch", "analysis_set", "model", "outcome", "N", "error"]].head(20).to_string(index=False))
                    (output / "reports/结果摘要.md").write_text(summary_markdown(summary, diet_result, micro_result, run_id), encoding="utf-8")
                    code = 0 if failures.empty else 2
                    try:
                        manifest["visualization"] = visuals.generate_visual_report(output, require_completed=False)
                        report = Path(manifest["visualization"]["report"]).relative_to(output / "visuals")
                        with (output / "reports/结果摘要.md").open("a", encoding="utf-8") as handle:
                            handle.write("\n## 直观统计与图表\n\n[打开结果总览](../visuals/%s)。包含统计表及七组图，每组提供 PNG、PDF、SVG。\n" % report.as_posix())
                    except Exception as error:
                        manifest["visualization"] = {"status": "failed", "error": "%s: %s" % (type(error).__name__, error)}
                        manifest["status"] = "completed_with_visualization_failure"
                        print("VISUALIZATION_ERROR=" + manifest["visualization"]["error"])
                        print("模型结果已保存；修复绘图依赖或错误后，可用 scripts/new_cgm_visuals.py --run-dir 单独补图。")
                        code = 3
                else:
                    manifest["status"] = "inputs_checked_no_models_fitted"
                    code = 0
                # Abort the completion claim if any source changed while running.
                if [data.fingerprint(item["path"]) for item in fingerprints] != fingerprints:
                    raise ValueError("Input files changed during execution; outputs are not a stable run")
                manifest["source_files_unchanged"] = True
            except Exception as error:
                manifest["status"] = "failed"
                manifest["error"] = "%s: %s" % (type(error).__name__, error)
                print("ERROR=" + manifest["error"], flush=True)
                code = 1
            finally:
                manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                print("STATUS=" + manifest["status"])
                print("OUTPUT_DIR=" + str(output))
                print("LOG_FILE=" + str(log_path))
                print("NOTE: 未自动进行路径/中介分析；本次 Z 与原值不重算。")
                with (output / "reports/manifest.json").open("w", encoding="utf-8") as handle:
                    json.dump(safe_json(manifest), handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.write("\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
