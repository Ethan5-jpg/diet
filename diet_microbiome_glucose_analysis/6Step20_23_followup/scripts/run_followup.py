#!/usr/bin/env python3
"""Step20-23: audited read-only analyses of the frozen HPP mediation results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd
from common import Context, KEYS, TAGS, DOMAINS, bools, num, need, unique, align, require, sha256, write_json
from cohort import load_population, audit_lineage, preprocessing_trace
from analyses import correlations, domain_analysis, methods_audit, mechanism_queue

PACKAGE = Path(__file__).resolve().parents[1]


def choose_step19b(ctx, explicit, plan):
    if explicit:
        return Path(explicit).expanduser().resolve()
    parent = ctx.root / "diet_microbiome_glucose_analysis/5Step19b_core_taxa_direction/outputs"
    known = parent / plan["known_step19b_run"]
    if (known / "manifest.json").is_file():
        return known
    candidates = []
    for file in sorted(parent.glob("step19b_*/manifest.json")):
        m = json.loads(file.read_text(encoding="utf-8"))
        if m.get("status") == "PASS" and m.get("data_kind") == "server_result_tables":
            candidates.append(file.parent)
    require(len(candidates) == 1, "Expected the known Step19b run or one unique PASS run; found {}. Specify --step19b-run. Candidates: {}".format(len(candidates), "; ".join(map(str, candidates))))
    return candidates[0]


def load_frozen(ctx, folder, plan):
    manifest = ctx.json(folder / "manifest.json", "step19b_manifest")
    require(manifest.get("status") == "PASS", "Step19b is not PASS")
    expected_kind = "SYNTHETIC_TEST_ONLY" if ctx.synthetic else "server_result_tables"
    require(manifest.get("data_kind") == expected_kind, "Synthetic/check-only Step19b data cannot be used as real results")
    require(manifest.get("all_inputs_unchanged_verified") is True, "Step19b did not verify unchanged inputs")
    source_paths = {}
    for record in manifest["inputs"]:
        p = ctx.record(record["path"], "step19b_input:" + record.get("label", "unknown"))
        require(sha256(p) == record["sha256"], "Changed Step19b input: " + str(p))
        source_paths[p.name] = p
    for record in manifest["outputs"]:
        p = ctx.record(folder / record["file"], "step19b_output")
        require(sha256(p) == record["sha256"], "Changed Step19b output: " + str(p))
    filename = "18f_threeway_path_robustness.csv"
    require(filename in source_paths, "Step19b manifest lacks frozen 18f source")
    paths = ctx.read(source_paths[filename], "frozen1490")
    high = ctx.read(folder / "19b_path_direction_details.csv", "retained871_directions")
    unique(paths, KEYS, "1490 candidates")
    unique(high, KEYS, "871 retained")
    for field in ["highest_robustness", "primary_retained_strict", "primary_retained_protocol"] + ["robust_" + t for t in TAGS]:
        need(paths, [field], "18f")
        paths[field] = bools(paths[field])
    primary, strict, protocol = [paths["robust_" + t] for t in TAGS]
    require(paths.highest_robustness.eq(primary & strict & protocol).all(), "Inconsistent three-way retained labels")
    require(paths.primary_retained_strict.eq(primary & strict).all(), "Invalid strict retention labels")
    require(paths.primary_retained_protocol.eq(primary & protocol).all(), "Invalid protocol retention labels")
    high = align(paths[paths.highest_robustness], high)
    e = plan["expected"]
    observed = dict(candidates=len(paths), retained=len(high), species=high.species.nunique(), primary=int(primary.sum()),
                    strict_retained=int(paths.primary_retained_strict.sum()), protocol_retained=int(paths.primary_retained_protocol.sum()))
    for key, value in observed.items():
        require(value == e[key], "Frozen count mismatch {}: {} != {}".format(key, value, e[key]))
    need(high, ["taxon_id", "display_species", "domain_breadth", "phenotype_domain"], "19b details")
    for domain, expected in e["domains"].items():
        require(int(high.phenotype_domain.eq(domain).sum()) == expected, "Frozen domain count mismatch: " + domain)
    models, metadata = {}, {}
    for tag in TAGS:
        p = ctx.relocate(manifest["selected_models"][tag])
        require(str(p) in ctx.inputs, "Selected model absent from Step19b verified input manifest")
        m = align(paths, ctx.read(p, "model_" + tag))
        need(m, ["indirect_effect", "N", "sample_sha256"], tag)
        for source, target in [("indirect_effect", "indirect_effect_" + tag), ("N", "N_" + tag)]:
            require(np.isclose(num(m[source]), num(paths[target]), atol=1e-12, rtol=1e-8, equal_nan=True).all(), "Model/18f value mismatch: " + tag + "/" + source)
        models[tag] = m
        meta = ctx.json(p.parent.parent / "reports/manifest.json", tag + "_model_manifest")
        require(meta.get("status") == "completed" and meta.get("debug_subset") is False, tag + " model manifest is not a complete run")
        metadata[tag] = meta
    ctx.csv(pd.DataFrame([observed]), "00_frozen_counts_verified.csv")
    return paths, high, models, metadata


def resolve_cgm(ctx, metadata, explicit):
    paths = [ctx.relocate(metadata[t]["final_cgm"]) for t in TAGS]
    if explicit:
        chosen = ctx.record(explicit, "explicit_CGM")
    else:
        chosen = ctx.record(paths[0], "primary_model_CGM")
    for p in paths:
        p = ctx.record(p, "model_CGM_reference")
        require(sha256(p) == sha256(chosen), "Three models reference different CGM file contents")
    # Historical 18b manifests record a path, not the phenotype file hash.
    ctx.notes.append("Model manifests identify CGM path but lack historical CGM hashes. Current value lineage and sample-hash equality do not prove historical numerical identity.")
    df = ctx.read(chosen, "CGM_for_followup")
    if not ctx.synthetic:
        require(len(df) == 7493, "Expected frozen CGM cohort N=7493")
    return df, chosen


def write_reports(ctx, plan, source_run):
    states = {s["stage"]: s for s in ctx.stages}
    bad = [s for s in ctx.stages if s["status"] != "PASS"]
    status = "PASS" if not bad else "REVIEW_REQUIRED"
    lines = ["=== HPP STEP20-23 FOLLOW-UP ===", "STATUS=" + status,
             "DATA_KIND=" + ("SYNTHETIC_TEST_ONLY" if ctx.synthetic else "SERVER_RESULT_TABLES"),
             "MODELS_REFIT=False | FDR_RECOMPUTED=False | FROZEN_SELECTION_CHANGED=False",
             "SDw=mean within-day SD | COGI=higher control score | GRADE_eugly=GRADE contribution, not TIR"]
    for stage in ctx.stages:
        lines.append(stage["stage"] + "=" + stage["status"] + (" | " + stage["error"] if "error" in stage else ""))
    lines += ["", "DEFINITION_VERIFIED != HISTORICAL_IMPLEMENTATION_VERSION_VERIFIED",
              "MECHANISM_REVIEW=structured evidence worklist; path-specific literature/functional validation remains open",
              "INTERPRETATION=descriptive cross-sectional associations; correlated paths are not independent evidence",
              "SOURCE_STEP19B=" + str(source_run), "OUTPUT_DIR=" + str(ctx.out), "RESULT_ARCHIVE=" + str(ctx.out / "aggregate_results.zip")]
    (ctx.out / "SCREENSHOT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for number, prefix in [("20", "20_"), ("21", "21_"), ("22", "22_"), ("23", "23_")]:
        section = ["=== STEP" + number + " ==="]
        for s in ctx.stages:
            if s["stage"].startswith(prefix):
                section.append(s["stage"] + "=" + s["status"])
                d = s.get("detail", {})
                if "error" in s:
                    section.append("  " + s["error"])
                if "full20" in d:
                    section.append("  DOMAIN | metrics | candidates | primary | both | both/candidate | both/primary")
                    for r in d["full20"]:
                        section.append("  {} | {} | {} | {} | {} | {:.1%} | {:.1%}".format(r["domain"], r["registered_metrics"], r["candidate_paths"], r["primary_robust"], r["retained_both"], r["both_over_candidates"], r["both_over_primary"]))
                elif "cohorts" in d:
                    for r in d["cohorts"]:
                        section.append("  {}: N={} | Mean/GMI pair N={} | rho={:.4f}".format(r["cohort"], r["N"], r["mean_gmi_pair_N"], r["mean_gmi_spearman"]))
                elif "summaries" in d:
                    for r in d["summaries"]:
                        section.append("  {}: within FDR<.05={} | global={} | retained with global<.05={} | N range={}-{}".format(r["model"], r["within_FDR05"], r["global_FDR05"], r["retained871_global_FDR05"], r["N_min"], r["N_max"]))
                elif d:
                    section.append("  " + json.dumps(d, ensure_ascii=False, default=str))
        (ctx.out / ("STEP" + number + "_SUMMARY.txt")).write_text("\n".join(section) + "\n", encoding="utf-8")
    manuscript = ["# 自动生成的结果核查与写作材料", "", "运行状态：" + status, "",
                  "此文件只有通过相应核查的汇总可以引用；REVIEW_REQUIRED时先解决对应阶段错误。所有结果为横断面统计关联。", ""]
    if "20_domains" in states and states["20_domains"]["status"] == "PASS":
        for r in states["20_domains"]["detail"]["full20"]:
            manuscript.append("- {}：{}项指标，{}条冻结候选，{}条primary稳健；双敏感性保留{}条。以候选为分母保留率{:.2%}，以primary稳健为分母保留率{:.2%}。".format(r["domain"], r["registered_metrics"], r["candidate_paths"], r["primary_robust"], r["retained_both"], r["both_over_candidates"], r["both_over_primary"]))
    manuscript += ["", "## 方法草稿", "",
                  "分析继承已冻结的7项饮食暴露、20项主要CGM结局和单菌中介候选；另5项CGM只用于补充描述。主模型基于既有糖尿病/A10排除标志不等于1的人群（保留该标志未知者）；严格分析仅纳入标志等于0者。具体纳入人数以逐路径完整病例数和样本哈希为准。",
                  "饮食指标沿用已处理的Z列，hPDI使用更正版本。菌群使用CLR后Z值，不能表述为绝对菌量。CGM沿用已有Z列，不在本步骤重新清洗或标准化。相关性按每对结局的共同非缺失观测计算，分别给出Pearson、Spearman及对应N；Spearman在每对完整病例内处理并列秩。",
                  "相关性基础人群要求主/严格纳入资格、CGM和菌群记录、至少一个饮食Z值及共同协变量完整，不要求所有20项结局完整，也不等同于任一中介路径样本。另提供进入至少一条冻结候选路径的参与者并集和全部CGM人群的描述性结果。",
                  "中介路径的主检验家族为各饮食×CGM结局内的冻结菌种集合，全局FDR仅继承原有结果作为敏感性；本包未重算FDR。抗生素/PPI敏感性使用饮食记录窗口字段，未将未知值填0。效应、95%bootstrap区间及N来自原模型。",
                  "领域保留率采用候选路径数和主模型稳健路径数两种明确分母，指标个数为3/9/8。去除GMI和概念简化面板均为事后描述性冗余检查，不是重新筛选的推断家族，不作独立路径的富集检验。",
                  "同一队列进行了路径筛选及后续分析，结果属于探索性证据；敏感性模型不构成独立验证。总效应较小时或总/间接效应方向不一致时不以中介比例作为主要结论；不同单菌路径的比例不能相加。", "",
                  "## 定义与待核实项", "",
                  "SDw的旧Interday标签应改为平均日内标准差；COGI升高是其设计上的较佳控制方向；GRADE eugly为目标范围对GRADE的贡献百分比。不得把所有间接效应正号归为不利，或把负号都归为有益。",
                  "公开文档核实不等于HPP历史计算版本、阈值和插值参数已被证实。字段raw匹配、clean→Z检查及旧清洗记录分别见核查表。源码中17项扩展配置未列SDw，服务器新增SDw的具体脚本/清洗参数仍需由实际来源记录补齐。",
                  "机制证据矩阵是需要逐项完成的研究表，空白不是没有机制。功能/代谢组文件盘点只证明文件存在。当前结果不能证明特定菌株产生某代谢物，也不能证明饮食通过该菌产生因果效应。", "",
                  "## 图表说明", "",
                  "20相关图按既有领域排序；空灰格表示样本不足或常量。20保留率图展示不等分母下的描述性分布。23方向图保留数学符号，空白表示不在871路径集合。23森林图展示原三模型bootstrap区间，独立横轴；示例按预定概念选取，含E. rectale的NOVA4/hPDI对应结局，不是挑最大效应。", "",
                  "## 运行核查", "", "```text", *lines, "```", "", "输入SHA256、软件版本、完整状态见manifest.json；详细路径、效应、分母和N见tables目录。"]
    (ctx.out / "reports/结果与方法草稿.md").write_text("\n".join(manuscript) + "\n", encoding="utf-8")
    return status, lines


def run(args, test_plan=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.output_parent).expanduser().resolve() / ("followup_" + stamp)
    for folder in ["tables", "reports", "figures", "logs"]:
        (out / folder).mkdir(parents=True, exist_ok=False)
    ctx = Context(Path(args.data_root).expanduser().resolve(), out, PACKAGE)
    ctx.synthetic = test_plan is not None
    plan = ctx.json(PACKAGE / "config/analysis_plan.json", "analysis_plan") if test_plan is None else test_plan
    dictionary = ctx.read(PACKAGE / "config/outcome_dictionary.csv", "outcome_dictionary")
    ctx.csv(dictionary, "20_dictionary_used.csv")
    source_run = None
    holder = {}

    def inputs():
        nonlocal source_run
        source_run = choose_step19b(ctx, args.step19b_run, plan)
        p, h, m, meta = load_frozen(ctx, source_run, plan)
        holder.update(paths=p, high=h, models=m, metadata=meta)
        return dict(candidates=len(p), retained=len(h), species=h.species.nunique(), source=str(source_run))

    ctx.stage("00_frozen_input_integrity", inputs)
    if holder:
        p, h, models = holder["paths"], holder["high"], holder["models"]
        ctx.stage("20_domains", lambda: domain_analysis(ctx, p, dictionary, plan, not args.no_plots))
        ctx.stage("21_original_models_FDR", lambda: methods_audit(ctx, p, models, plan))

        def phenotype():
            df, path = resolve_cgm(ctx, holder["metadata"], args.cgm_csv)
            holder["cgm"] = df
            holder["cgm_path"] = path
            return dict(N=len(df), source=str(path))

        ctx.stage("20_CGM_source", phenotype)
        if "cgm" in holder:
            ctx.stage("21_preprocessing_trace", lambda: preprocessing_trace(ctx, holder["cgm"], holder["cgm_path"], dictionary.field.tolist()))
            def lineage():
                raw = Path(args.iglu_csv).expanduser().resolve() if args.iglu_csv else ctx.root / "Transfer/cgm/iglu.csv"
                result = audit_lineage(ctx, holder["cgm"], dictionary.field.tolist(), raw)
                holder["lineage"] = result
                ok = result.raw_lineage_status.eq("PASS") & result.clean_to_z_status.eq("PASS")
                status = "PASS" if ok.all() else "REVIEW_REQUIRED"
                return dict(status=status, verified=int(ok.sum()), total=len(result), needs_review=result.loc[~ok, ["outcome_field", "raw_lineage_status", "clean_to_z_status"]].to_dict("records"))

            ctx.stage("20_field_lineage", lineage)

            def population():
                cg, masks, detail = load_population(ctx, holder["cgm"], models)
                holder.update(cg=cg, masks=masks)
                return detail

            ctx.stage("21_sample_N_and_hash", population)
            lineage_ok = "lineage" in holder and (holder["lineage"].raw_lineage_status.eq("PASS") & holder["lineage"].clean_to_z_status.eq("PASS")).all()
            if "cg" in holder and lineage_ok:
                ctx.stage("20_correlations", lambda: correlations(ctx, holder["cg"], holder["masks"], dictionary, plan, not args.no_plots))
            else:
                ctx.stages.append(dict(stage="20_correlations", status="SKIPPED_DEPENDENCY", error="Sample reconstruction or field lineage/Z audit has not passed"))
        else:
            ctx.stages.append(dict(stage="21_sample_N_and_hash", status="SKIPPED_DEPENDENCY", error="CGM source unavailable"))
            ctx.stages.append(dict(stage="20_correlations", status="SKIPPED_DEPENDENCY", error="CGM source unavailable"))
        ctx.stage("22_mechanism_materials", lambda: mechanism_queue(ctx, h, dictionary))
        if args.no_plots:
            ctx.stages.append(dict(stage="23_figures", status="SKIPPED_EXPLICIT", detail={"no_plots": True}))
        else:
            from plots import core_figures
            figure_dictionary = dictionary.copy()
            verified = set()
            if "lineage" in holder:
                verified = set(holder["lineage"].loc[holder["lineage"].raw_lineage_status.eq("PASS") & holder["lineage"].clean_to_z_status.eq("PASS"), "outcome_field"])
            for f in ["cgm_sdw", "cgm_cogi", "cgm_grade_eugly"]:
                if f not in verified:
                    figure_dictionary.loc[figure_dictionary.field.eq(f), "label"] = f + " [lineage pending]"
            ctx.stage("23_figures", lambda: core_figures(ctx, h, figure_dictionary))
    for name in ["官方定义核查与解释修正.md", "方法核对及后续写作边界.md", "机制证据起点与研究清单.md", "public_source_register.json"]:
        file = ctx.record(PACKAGE / "docs" / name, "packaged_research_notes")
        shutil.copyfile(file, out / "reports" / name)
    ctx.stage("99_inputs_unchanged", lambda: ctx.unchanged())
    status, lines = write_reports(ctx, plan, source_run)
    import matplotlib
    code = [p for folder in ["scripts", "config", "docs"] for p in (PACKAGE / folder).rglob("*") if p.is_file() and "__pycache__" not in str(p)]
    manifest = dict(status=status, data_kind="SYNTHETIC_TEST_ONLY" if ctx.synthetic else "SERVER_RESULT_TABLES",
                    utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
                    models_refit=False, fdr_recomputed=False, plan=plan, parameters=vars(args), inputs=list(ctx.inputs.values()),
                    stages=ctx.stages, notes=ctx.notes, code_and_config=[dict(file=str(p.relative_to(PACKAGE)), sha256=sha256(p)) for p in code])
    manifest["outputs"] = [dict(file=str(p.relative_to(out)), sha256=sha256(p)) for p in sorted(out.rglob("*")) if p.is_file()]
    write_json(out / "manifest.json", manifest)
    # This archive contains only aggregated tables/figures/audit metadata, no participant matrices.
    with zipfile.ZipFile(out / "aggregate_results.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.name != "aggregate_results.zip":
                archive.write(p, str(p.relative_to(out)))
    write_json(Path(args.output_parent) / "LAST_RUN.json", dict(run_dir=str(out), status=status))
    print("\n" + "\n".join(lines), flush=True)
    return (0 if status == "PASS" else 2), out


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=str(PACKAGE.parents[1]), help="HPP Data directory; inferred from this package location")
    p.add_argument("--step19b-run", help="Exact Step19b output directory; default is confirmed run")
    p.add_argument("--cgm-csv", help="Explicit phenotype path; must match model-referenced contents")
    p.add_argument("--iglu-csv", help="Official iglu CSV export; default Data/Transfer/cgm/iglu.csv")
    p.add_argument("--output-parent", default=str(PACKAGE / "outputs"))
    p.add_argument("--no-plots", action="store_true", help="Diagnostic tables only; overall status records skipped figures")
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args())[0])
