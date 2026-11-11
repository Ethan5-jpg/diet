"""Compact aggregate reports for current and archived run artifacts."""

from exp2_engine import BASE, WORKFLOWS


POOL_LABELS = {
    "select_core_other": "select_18_other",
    "select_v2_other": "select_34_other",
    "select_v1_release_other": "select_44_other",
    "select_full": "all 89 extended",
    "select_full_guarded": "guarded 89",
    "select_added": "latest 18 only",
}


def label(key):
    return POOL_LABELS.get(key, key)


def pm(value):
    return "%+.6f" % value


def interval(bounds):
    return "[%s,%s]" % tuple(map(pm, bounds))


def maybe_pm(value):
    return "NA" if value is None else pm(value)


def expanded_overview(summary, s0, s1):
    followup = bool(summary.get("followup_routes"))
    if followup:
        keys = ("select_ridge", "select_other", "select_expanded", "select_added",
                "select_full", "select_full_guarded")
    else:
        keys = ("select_ridge", "select_other", "select_new", "select_expanded", "select_guarded",
                "select_all", "blend")
    names = {"select_other": "old 53 extended", "select_new": "new 18 only",
             "select_expanded": "all 71 extended", "select_guarded": "guarded 71",
             "select_all": "old 65 all", "blend": "old 65 blend"}
    print("workflow          full_R2_0 full_R2_1   dR2_0      dR2_1      vs_ridge0  vs_ridge1")
    for key in keys:
        a, b = s0[key], s1[key]
        print("%-17s %9.6f %9.6f %s %s %s %s" %
              (names.get(key, label(key)), a["r2"], b["r2"], pm(a["delta"]), pm(b["delta"]),
               pm(a["vs_ridge"]), pm(b["vs_ridge"])))
    primary = "select_full" if followup else "select_expanded"
    a, b = s0[primary], s1[primary]
    print("Primary workflow uses %d extended routes, repeat 0. dR2 compares against common main." %
          (89 if followup else 71))
    comparisons = [("vs common main", "delta", "ci"), ("vs ridge", "vs_ridge", "vs_ridge_ci"),
                   ("vs old 53 extended", "vs_legacy", "vs_legacy_ci")]
    if followup:
        comparisons.append(("vs old 71 extended", "vs_previous", "vs_previous_ci"))
    for text, key, ci_key in comparisons:
        print("%-18s r0=%s %s  r1=%s %s" %
              (text, pm(a[key]), interval(a[ci_key]), pm(b[key]), interval(b[ci_key])))
    print("Expanded dR2 meets in both repeats: " +
          " | ".join("%.1f pp %s" % (100 * target, "yes" if hit else "no")
                     for target, hit in zip(summary["thresholds"], summary["targets"][primary])))


def expanded_diagnostics(summary, s0, s1, q0, q1):
    followup = bool(summary.get("followup_routes"))
    primary = "select_full" if followup else "select_expanded"
    guard = "full_guard" if followup else "guard"
    keys = (("select_ridge", "select_expanded", "select_added", "select_full", "select_full_guarded")
            if followup else ("select_ridge", "select_other", "select_new", "select_expanded", "select_guarded"))
    print("Local 10->15 range support: N=%d/%d in-range, N=%d/%d outside (r0/r1)." %
          tuple(q["support_strata"][key]["people"] for key in ("supported", "outside") for q in (q0, q1)))
    print("workflow          rawY dR2_0/1       in-range dR2_0/1   outside dR2_0/1  split_corr")
    for key in keys:
        a, b = s0[key], s1[key]
        corr = summary["contrast_correlation"][key]
        print("%-17s %s/%s  %s/%s  %s/%s  %s" %
              (label(key), pm(a["raw_y_delta"]), pm(b["raw_y_delta"]),
               maybe_pm(a["supported"]["delta"]), maybe_pm(b["supported"]["delta"]),
               maybe_pm(a["outside"]["delta"]), maybe_pm(b["outside"]["delta"]),
               "NA" if corr is None else "%.3f" % corr))
    print("Primary 10->15 correction contrast Q10/Q50/Q90 (mg/dL), supported people:")
    for i, s in enumerate((s0, s1)):
        q = s[primary]["supported"]["contrast_quantiles"]
        print("r%d %s" % (i, "NA" if q is None else "/".join("%+.3f" % x for x in q)))
    print("Guard passed in %d/%d folds r0/r1; paired inner household rule, no p-value." %
          tuple(sum(a["passed"] for a in q[guard]) for q in (q0, q1)))
    for i, q in enumerate((q0, q1)):
        print("r%d expanded choices: %s" %
              (i, ", ".join(m[primary] or "null" for m in q["selected"])))
    print("Range strata are descriptive; both remain in the primary score.")


def route_table(routes, s0, s1, heading):
    print(heading + " | dR2 = model R2 - common main R2")
    applied = all("applied_folds" in s0[key] for key in routes)
    count = "applied_folds" if applied else "active_folds"
    print("route                   delta0     delta1     %s  corr_SD0/1" %
          ("used folds" if applied else "chosen+ folds"))
    for key in routes:
        a, b = s0[key], s1[key]
        print("%-23s %s %s     %d/%d          %.3f/%.3f" %
              (key, pm(a["delta"]), pm(b["delta"]), a[count], b[count],
               a["contrast_sd"], b["contrast_sd"]))
    if applied:
        print("Used folds count corrections actually applied in outer predictions.")
    else:
        print("chosen+ counts active settings chosen in inner CV.")


def dose_replay_report(summary, s0, s1):
    print("Re-evaluate fitted correction with fiber fixed at %g g/1000 kcal." % summary["contrast"][0])
    print("workflow             full dR2_0/1       fixed only_0/1     dose varying_0/1    full vs fixed_0/1")
    for key in summary["anchor_workflows"]:
        a, b = s0[key], s1[key]
        values = [a["delta"], b["delta"]]
        for field in ("fixed_only", "varying_only", "full_vs_fixed"):
            values.extend(s["dose_replay"][field] for s in (a, b))
        print("%-19s %s/%s  %s/%s  %s/%s  %s/%s" % (label(key), *map(pm, values)))
    print("Fixed and varying components share the same fitted correction. Scores are descriptive.")


def display(summary, screen=1, synthetic=False):
    s0, s1 = summary["scores"]
    q0, q1 = summary["quality"]
    workflows = summary.get("workflows", [name for name in WORKFLOWS if name in s0])
    added = summary.get("added_routes", [])
    additional = summary.get("v3_routes", [])
    release_routes = summary.get("release_added_routes", [])
    expansion = summary.get("expansion_routes", [])
    followup = summary.get("followup_routes", [])
    core = summary.get("core_routes", [name for name in summary["names"]
                                      if name != BASE and name not in workflows])
    print("%s | %s | %s -> %s (%s) | N=%d HH=%d" %
          ("SYNTHETIC" if synthetic else "DISCOVERY", summary["version"], summary["exposure"],
           summary["outcome"], summary["unit"], q0["people"], q0["households"]))
    print("Matched common main R2: r0=%.6f r1=%.6f; r1 re-splits the same people." %
          (s0[BASE]["r2"], s1[BASE]["r2"]))
    if screen == 1 and expansion:
        expanded_overview(summary, s0, s1)
    elif screen == 1:
        print("workflow          full_R2_0 full_R2_1   delta0     delta1       fixed-OOF HH95CI0")
        for key in workflows:
            a, b = s0[key], s1[key]
            print("%-17s %9.6f %9.6f %s %s %s" %
                  (label(key), a["r2"], b["r2"], pm(a["delta"]), pm(b["delta"]), interval(a["ci"])))
        print("workflow          vs_ridge0  vs_ridge1     fixed-OOF HH95CI0 vs ridge")
        for key in workflows:
            if key != "select_ridge":
                print("%-17s %s %s %s" %
                      (label(key), pm(s0[key]["vs_ridge"]), pm(s1[key]["vs_ridge"]), interval(s0[key]["vs_ridge_ci"])))
        if added:
            a, b = s0["select_other"], s1["select_other"]
            print("Full vs 18-route other pool: r0=%s r1=%s HH95CI0=%s" %
                  (pm(a["vs_core"]), pm(b["vs_core"]), interval(a["vs_core_ci"])))
        if additional:
            a, b = s0["select_other"], s1["select_other"]
            print("Full vs 34-route other pool: r0=%s r1=%s HH95CI0=%s" %
                  (pm(a["vs_v2"]), pm(b["vs_v2"]), interval(a["vs_v2_ci"])))
        if release_routes:
            a, b = s0["select_other"], s1["select_other"]
            print("Full vs 44-route other pool: r0=%s r1=%s HH95CI0=%s" %
                  (pm(a["vs_v1_release"]), pm(b["vs_v1_release"]), interval(a["vs_v1_release_ci"])))
        print("Thresholds in BOTH repeats: " + ", ".join("%.1f pp" % (100 * t) for t in summary["thresholds"]))
        for start in range(0, len(workflows), 3):
            print(" | ".join("%s %s" % (label(key), "/".join("yes" if ok else "no" for ok in summary["targets"][key]))
                             for key in workflows[start:start + 3]))
        print("Primary workflow: select_other, repeat 0.")
    elif screen == 2:
        route_table(core, s0, s1, "Base routes (26)" if added else "All routes")
    elif screen == 3 and added:
        route_table(added, s0, s1, "Extended routes (20)")
        print("Supported range r0/r1: %.1f%%/%.1f%%" % (100 * q0["support"], 100 * q1["support"]))
        print("Inner failed-fit events r0/r1: %d/%d | Outer zero fallbacks: %d/%d" % tuple(
            sum(v for k, v in q["status"].items() if k.startswith(prefix))
            for prefix in ("failed_", "outer_failed_") for q in (q0, q1)))
        print("Full stability and fit diagnostics: --screen 4")
    elif screen == 9:
        if followup:
            route_table(followup, s0, s1, "Latest six families (%d routes)" % len(followup))
        else:
            print("The latest route group is unavailable for this run.")
    elif screen == 10:
        if summary.get("anchor_workflows"):
            dose_replay_report(summary, s0, s1)
        else:
            print("Fixed-dose replay is unavailable for this run.")
    elif screen == 7:
        if expansion:
            route_table(expansion, s0, s1, "Six added families (%d routes)" % len(expansion))
        else:
            print("No expansion routes in this run.")
    elif screen == 8:
        if expansion:
            expanded_diagnostics(summary, s0, s1, q0, q1)
        else:
            print("Use --screen 4 for this run's diagnostics.")
    elif screen == 6:
        if release_routes:
            route_table(release_routes, s0, s1, "Final route group (9)")
            a, b = s0["select_other"], s1["select_other"]
            print("Original 53-route pool vs 44-route pool: r0=%s r1=%s HH95CI0=%s" %
                  (pm(a["vs_v1_release"]), pm(b["vs_v1_release"]), interval(a["vs_v1_release_ci"])))
        else:
            print("This release's additional routes are on screen 5.")
    elif screen == 5:
        if additional:
            route_table(additional, s0, s1, "Additional routes (10)")
            a, b = summary["joint_comparison"]
            print("Oral + gut + tensor vs oral + gut slopes")
            print("r0=%s r1=%s HH95CI0=%s" %
                  (pm(a["delta"]), pm(b["delta"]), interval(a["ci"])))
            print("Blend solver fallback folds r0/r1: %d/%d" %
                  (q0["status"].get("blend_fallback", 0), q1["status"].get("blend_fallback", 0)))
        else:
            print("Use --screen 2 and --screen 3 for this edition's route tables.")
    else:
        print("Contrast: 10 -> 15 g fiber/1000 kcal, added correction in mg/dL.")
        print("Local range support: r0=%.1f%% r1=%.1f%%." % (100 * q0["support"], 100 * q1["support"]))
        print("workflow           rawY dR2_0  rawY dR2_1  corr_SD0/1(mg/dL)   split_corr")
        for key in workflows:
            a, b = s0[key], s1[key]
            cor = summary["contrast_correlation"][key]
            print("%-18s %s   %s   %.3f/%.3f           %s" %
                  (label(key), pm(a["raw_y_delta"]), pm(b["raw_y_delta"]), a["contrast_sd"],
                   b["contrast_sd"], "NA" if cor is None else "%.3f" % cor))
        if additional:
            print("workflow           abs_correction_p99/maximum r0 and r1 (mg/dL)    worst_fold_delta0/1")
            for key in workflows:
                a, b = s0[key], s1[key]
                print("%-18s %6.2f/%-6.2f  %6.2f/%-6.2f     %s/%s" %
                      (label(key), a["correction_p99"], a["correction_max"], b["correction_p99"], b["correction_max"],
                       *(pm(s["worst_fold_delta"]) if s["worst_fold_delta"] is not None else "NA" for s in (a, b))))
        for i, q in enumerate((q0, q1)):
            print("r%d projection ranks=%s warnings=%d total runtime %.1f min" %
                  (i, q["projected_ranks"], len(q["warnings"]), q["seconds"] / 60))
            if "max_projected_condition" in q:
                print("r%d maximum retained condition %.1f" % (i, q["max_projected_condition"]))
            print("r%d diagnostics=%s" % (i, q["status"]))
            if q.get("failed_routes"):
                print("r%d routes with numerical failures: %s" % (i, ", ".join(q["failed_routes"])))
            for key in workflows:
                if key in q["selected"][0]:
                    print("r%d %-17s %s" % (i, label(key), ", ".join(x[key] or "null" for x in q["selected"])))
        print("Raw Y uses unchanged predictions; split correlation measures sensitivity.")
    print("HH95CI resamples households with fitted predictions fixed.")
