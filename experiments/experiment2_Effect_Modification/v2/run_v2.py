"""Exact-V11 person-level fiber by microbiome modification."""

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
FILES = {
    "person_inputs.py", "moderation_models.py", "marker_effects.py", "dose_sensitivity.py",
    "selftest.py", "run_v2.py", "server_run.sh",
}
VERSIONS = {
    "numpy": "1.26.4", "scipy": "1.11.4", "pandas": "2.1.0", "pyarrow": "15.0.2",
    "scikit-learn": "1.4.1.post1", "threadpoolctl": "3.3.0",
}


def out(message):
    print(message, flush=True)


def check_release():
    manifest = json.loads((ROOT / "upload_manifest.json").read_text(encoding="utf-8"))["files"]
    if set(manifest) != FILES:
        raise ValueError("UPLOAD_FILE_SET_CHANGED")
    for file, expected in manifest.items():
        if sha256((ROOT / file).read_bytes()).hexdigest() != expected:
            raise ValueError("UPLOAD_HASH_MISMATCH " + file)
    out(f"EXP2_V2_UPLOAD_OK files={len(FILES)}")
    try:
        found = {name: version(name) for name in VERSIONS}
    except PackageNotFoundError as exc:
        raise ValueError("HPP_ENV_MISSING " + exc.name) from None
    different = [f"{name}={found[name]} (need {pin})" for name, pin in VERSIONS.items() if found[name] != pin]
    if different:
        raise ValueError("HPP_ENV_MISMATCH " + ", ".join(different))
    out("EXP2_V2_ENV_OK")


def report_association(repeat, effects, *, raw=False):
    tag = "EXP2_V2_RAW_EFFECT" if raw else "EXP2_V2_EFFECT"
    out(f"{tag} r{repeat} | 10->15 g/1000kcal; Q75-Q25 in mg/dL")
    for name, effect in effects["markers"].items():
        support = ""
        if not raw:
            s = effect["support"]
            low, high = s["low"], s["high"]
            covered = "yes" if s["both_tails_cover_10_to_15"] else "no"
            support = (f" tails_Q05-95=[{low['q05']:.1f},{low['q95']:.1f}]/"
                       f"[{high['q05']:.1f},{high['q95']:.1f}] covers={covered} "
                       f"near10/15={low['near_10']}/{low['near_15']},{high['near_10']}/{high['near_15']}")
        if effect["status"] != "ok":
            out(f"  {name:17s} unavailable ({effect['reason']}){support}")
            continue
        a, b = effect["family95_mg_dl"]
        line = f"  {name:17s} effect={effect['estimate_mg_dl']:+.3f} family95=[{a:+.3f},{b:+.3f}]"
        if not raw:
            lo, hi = effect["ci95_mg_dl"]
            line += f" CI95=[{lo:+.3f},{hi:+.3f}]"
        out(line + support)


def report_dose(repeat, result):
    out(f"EXP2_V2_DOSE_SHAPE r{repeat} | gut Shannon Q75-Q25; fixed knot at 12.5 g/1000kcal")
    tails = result["support"]
    out(f"  near12.5 (within 1 g/1000kcal) low/high={tails['low']['near_hinge']}/{tails['high']['near_hinge']}")
    if result["status"] != "ok":
        out(f"  unavailable ({result['reason']})")
        return
    for name, contrast in result["contrasts"].items():
        interval = contrast.get("dose_g_per_1000_kcal")
        label = f"{interval[0]:g}->{interval[1]:g}" if interval else name
        lo, hi = contrast["ci95_mg_dl"]
        out(f"  {label} effect={contrast['estimate_mg_dl']:+.3f} mg/dL "
            f"pointwise_CI95=[{lo:+.3f},{hi:+.3f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeat", type=int, choices=(0, 1))
    parser.add_argument("--host-sensitivity", action="store_true",
                        help="Allow age and sex to interact with microbiome in the prediction baseline.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    check_release()

    import numpy as np
    from threadpoolctl import threadpool_limits
    from dose_sensitivity import dose_fold, summarize_dose
    from marker_effects import MARKERS, partial_fold, summarize_effects
    from moderation_models import OPTIONS, SEED, ZERO, run_fold, scores
    from person_inputs import SourceError, load_discovery

    stage = "preflight" if args.preflight else "frozen V11 inputs"
    try:
        with threadpool_limits(limits=args.threads):
            if args.preflight:
                from selftest import selftest

                selftest()
                out("EXP2_V2_PREFLIGHT_DONE | generated data")
                return 0
            source = load_discovery(ROOT)
            included = np.isfinite(source.reference["y"][0])
            n = np.count_nonzero(included)
            g = len(np.unique(source.groups[included]))
            fiber = source.index.fiber_density.to_numpy(dtype=float, na_value=np.nan)
            host_sensitivity = args.host_sensitivity
            options = {key: OPTIONS[key] for key in ("gut_shannon", "learned_both")} if host_sensitivity else OPTIONS
            predict_tag = "EXP2_V2_HOST_PREDICT" if host_sensitivity else "EXP2_V2_PREDICT"
            stability_tag = "EXP2_V2_HOST_STABILITY" if host_sensitivity else "EXP2_V2_STABILITY"
            gut_shannon = MARKERS.index("gut_shannon")
            out(f"EXP2_V2_SOURCE_READY people={n} households={g}")
            out("EXP2_V2_V11_REFERENCE r0=%.6f r1=%.6f" % source.historical_scores)
            contrast_history = {}
            repeats = (0, 1) if args.repeat is None else (args.repeat,)
            for repeat in repeats:
                stage = f"person-level repeat {repeat}"
                parts, effects, dose_parts = [], [], []
                for fold in range(5):
                    role, base, full, contrast, chosen, base_setting = run_fold(
                        source, repeat, fold, options=options, shared_host=host_sensitivity,
                    )
                    y = role.y[1]
                    if not host_sensitivity:
                        exposure = fiber[role.rows]
                        markers = role.raw_eco[1]
                        effects.append(partial_fold(
                            role.rows, y, role.association_main, exposure, markers, raw_y=source.y[role.rows],
                        ))
                        dose_parts.append(dose_fold(
                            role.rows, y, role.association_main, exposure, markers[:, gut_shannon],
                        ))
                    parts.append((role.rows, y, base, full, contrast, chosen))
                    shared = f" shared_age_sex_columns={len(role.shared_columns)}" if host_sensitivity else ""
                    out(f"EXP2_V2_OUTER_COMPLETE r{repeat} f{fold} people={len(y)} baseline={base_setting}{shared}")
                rows, y, base, full, contrast = (np.concatenate([p[k] for p in parts]) for k in range(5))
                order = np.argsort(rows)
                reference = source.reference["y"][repeat]
                expected = np.flatnonzero(np.isfinite(reference))
                if not np.array_equal(rows[order], expected) or not np.array_equal(y[order], reference[expected]):
                    raise SourceError("Pooled evaluated rows or D_Glu targets differ from V11")
                base_r2, full_r2, delta, ci = scores(
                    y[order], base[order], full[order], source.groups[expected], seed=SEED + repeat,
                )
                source.verify_sources()
                if not host_sensitivity:
                    report_association(repeat, summarize_effects(effects, source.groups))
                    report_association(repeat, summarize_effects(effects, source.groups, raw=True), raw=True)
                    report_dose(repeat, summarize_dose(dose_parts, source.groups))
                out(f"{predict_tag} r{repeat} N={n} baseline_R2={base_r2:+.6f}")
                out("  route                 full_R2   delta_R2      household_CI95    active")
                for j, name in enumerate(options):
                    lo, hi = ci[j]
                    used = sum(p[5][name] != ZERO for p in parts)
                    out(f"  {name:19s} {full_r2[j]:+.6f} {delta[j]:+.6f} "
                        f"[{lo:+.6f},{hi:+.6f}] {used}/5")
                contrast_history[repeat] = contrast[order]
            if len(contrast_history) == 2:
                out(f"{stability_tag} | same people; fitted 10->15 added contrast")
                for j, name in enumerate(options):
                    a, b = (contrast_history[k][:, j] for k in (0, 1))
                    s0, s1 = a.std(), b.std()
                    corr = f"{np.corrcoef(a, b)[0, 1]:+.3f}" if min(s0, s1) > 1e-10 else "NA"
                    out(f"  {name:19s} contrast_SD0/1={s0:.3f}/{s1:.3f} mg/dL corr={corr}")
            scope = "age/sex baseline sensitivity" if host_sensitivity else "primary and dose shape"
            repeats_run = ",".join(f"r{repeat}" for repeat in repeats)
            out(f"EXP2_V2_COMPLETE | {scope}; {repeats_run}; discovery only")
    except SourceError as exc:
        print(f"EXP2_V2_STOP stage={stage} reason={exc}", file=sys.stderr, flush=True)
        return 1
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, AssertionError) as exc:
        detail = str(exc)
        reason = detail if detail.isascii() and detail.replace("_", "").isupper() and " " not in detail else type(exc).__name__
        print(f"EXP2_V2_STOP stage={stage} error={reason}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"EXP2_V2_STOP release: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
