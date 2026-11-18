"""Fit the fixed three-species surface and nested alternatives on disjoint dates."""
import os

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import re
import sys

import numpy as np
from scipy.stats import t

import surface

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent.parent
DATES = ROOT.parent / "diet_fiber_date_split_20261113_v1"
PANEL = PACKAGE / "experiment2_Effect_Modification/v3/fermentation_score.py"
if not PANEL.exists():
    PANEL = ROOT.parent / "experiment2_Effect_Modification/v3/fermentation_score.py"
DIRECTIONS = ("early_to_late", "late_to_early", "all_same")
SEED = 2026111801
DATE_COMMIT = "7dc6be249b3ca1155d6040669444e2d3a4fef499"


def module(name, path):
    spec = spec_from_file_location(name, path)
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


sys.path.insert(0, str(DATES))
try:
    dates = module("hpp_split_dates", DATES / "run.py")
finally:
    sys.path.pop(0)
panel = module("hpp_fixed_panel", PANEL)
prior = dates.prior
digest, read, require, io = dates.digest, dates.read_json, dates.require, dates.io


def parts(smoke):
    return [(0, 0)] if smoke else [(r, f) for r in range(2) for f in range(5)]


def fold_results(roles, eco, scores, family, held, pcs_for, seed):
    train = ~held
    require(not set(family[held]) & set(family[train]), "Outer household overlap")
    losses = {key: defaultdict(float) for key in roles}
    for it, iv in surface.inner_roles(family, train, seed):
        pcs = pcs_for(it)
        for key, role in roles.items():
            data = surface.design(role, eco, pcs, scores, it)
            for route, pred in surface.candidates(*data, it, iv).items():
                losses[key][route] += float(np.sum((data[0][iv] - pred) ** 2))
    pcs = pcs_for(train)
    totals, selected = {}, {}
    for key, role in roles.items():
        data = surface.design(role, eco, pcs, scores, train)
        y = data[0]
        all_preds = surface.candidates(*data, train, held)
        tuned, joint = (surface.choose(losses[key], name) for name in ("panel", "joint"))
        selected[key] = dict(tuned=tuned, joint=joint)
        routes = [("baseline", None), ("shannon", (.03, .03)), ("panel", surface.FIXED), tuned, joint]
        errors = [(y[held] - y[train].mean()) ** 2] + [(y[held] - all_preds[k]) ** 2 for k in routes]
        totals[key] = np.column_stack([np.bincount(family[held], weights=a, minlength=family.max()+1) for a in errors])
    return totals, selected


def score(totals, weights):
    den, *losses = (weights @ totals).T
    require(np.all(np.isfinite(den) & (den > 0)), "Invalid outcome variance")
    base, _, fixed, _, _ = losses
    rbase = 1 - base[0] / den[0]
    result = {}
    for route, loss in zip(surface.ROUTES, losses):
        delta = (base - loss) / den
        paired = (fixed - loss) / den
        result[route] = dict(r2=float(1-loss[0]/den[0]), delta_r2=float(delta[0]),
                             relative_pct=float(100*delta[0]/rbase) if rbase > 0 else None,
                             delta_ci95=np.quantile(delta[1:], [.025, .975]).tolist(),
                             vs_fixed=float(paired[0]), vs_fixed_ci95=np.quantile(paired[1:], [.025, .975]).tolist())
    return result


def prediction(out, reg):
    fingerprint = digest(out / "registration.json")
    count = reg["flow"]["households"]
    result = {}
    for r in sorted({r for r, f in parts(reg["smoke"])}):
        totals = {k: np.zeros((count, 1+len(surface.ROUTES))) for k in DIRECTIONS}
        for rr, f in parts(reg["smoke"]):
            if rr != r:
                continue
            path = out / f"r{r}_f{f}.npz"
            meta = read(path.with_suffix(".json"))
            require(meta["sha"] == digest(path) and meta["registration"] == fingerprint, "Fold receipt mismatch")
            with np.load(path, allow_pickle=False) as payload:
                require(set(payload.files) == set(totals), "Unexpected directions")
                for key in totals:
                    totals[key] += payload[key]
        rng = np.random.default_rng(SEED+r)
        weights = np.ones((2000, count))
        for row in weights[1:]:
            row[:] = np.bincount(rng.integers(count, size=count), minlength=count)
        result[f"r{r}"] = {key: score(value, weights) for key, value in totals.items()}
    return result


def direct(roles, eco, pcs, scores, family, count):
    train = np.ones(len(family), dtype=bool)
    a, b = np.quantile(scores[:, 0], [.25, .75])
    m = np.median(scores[:, 0])
    marker_diff = np.array([b-a, max(b-m, 0)-max(a-m, 0)])/scores[:, 0].std()
    result = {}
    for key, role in roles.items():
        y, x, z, _ = surface.design(role, eco, pcs, scores, train)
        dose = role[2]
        slope = np.linalg.lstsq(np.column_stack((np.ones(len(y)), dose[:, 0])), dose[:, 1], rcond=1e-10)[0][1]
        contrast = np.outer([1, .5-slope], marker_diff).ravel()
        fit = dates.infer(y, x, z, train.astype(float), family, count, 0)
        result[key] = dates.effect(fit, contrast, 1, t.ppf(.975, count-1))
    return result


def verify(out, commit):
    reg = read(out / "registration.json")
    require(reg["commit"] == commit, "Release mismatch")
    require(all(digest(Path(p)) == h for p, h in reg["sources"].items()), "Source changed")
    names = ["registration.json", "summary.json", "stage_record.md"]
    names.extend(f"r{r}_f{f}.json" for r, f in parts(reg["smoke"]))
    require(read(out / "COMPLETE.json") == {n: digest(out/n) for n in names}, "Receipt mismatch")
    require(prediction(out, reg) == read(out / "summary.json")["prediction"], "Prediction readback mismatch")
    io.emit("SURFACE_READBACK_PASS", folds=len(parts(reg["smoke"])))


def stage(summary, reg):
    flow = reg["flow"]
    lines = ["# Three-species fiber surface on split dates", "", f"Release {reg['commit']}",
             f"People {flow['people']}, households {flow['households']}, paired days {flow['days']}.",
             "Relative percentage equals 100 times the paired R2 gain divided by positive baseline R2.",
             "Intervals use a paired household bootstrap conditional on fitted out-of-fold predictions.", "",
             "| Repeat | Dates | Model | R2 | R2 gain | Relative % | Gain vs fixed (95% CI) |",
             "|---|---|---|---:|---:|---:|---|"]
    for r, directions in summary["prediction"].items():
        for key, routes in directions.items():
            for name, metric in routes.items():
                pct = metric["relative_pct"]
                a, b = metric["vs_fixed_ci95"]
                lines.append(f"| {r} | {key} | {name} | {metric['r2']:.7f} | {metric['delta_r2']:.7f} | "
                             f"{pct if pct is not None else 'undefined'} | {metric['vs_fixed']:.7f} ({a:.7f}, {b:.7f}) |")
    lines += ["", "Three-species Q75 versus Q25 difference in the 10 to 15 g/1000 kcal fiber contrast (mg/dL)"]
    for key, v in summary["direct"].items():
        lines.append(f"- {key} {v['mg_dl']:.4f}, 95% CI {v['ci95']}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    require(re.fullmatch(r"[0-9a-f]{40}", args.commit) and re.fullmatch(r"[a-zA-Z0-9_-]+", args.run), "Bad run ID")
    out = ROOT / "runs_TRE_ONLY" / args.run
    if args.verify:
        verify(out, args.commit)
        return
    require(not out.exists(), "Use a fresh run ID")
    pins = {DATES/"run.py":"57a2936ffd13352812ebee3a237ea789eab1b7bae6b3f071176a15f76409734e",
            DATES/"date_split.py":"2ab12192177315814ca2850bfcaf03f06ba4be81f2131ac75cd3010371eff5bb",
            PANEL:"87950719455e3b7678ad942b4abef211ab47b79f1bca55b051e7174ccddbe6ef"}
    require(all(digest(k) == v for k,v in pins.items()), "Date or panel source pin differs")
    prev = DATES / "runs_TRE_ONLY/discovery-1"
    dates.verify(prev, DATE_COMMIT)
    old = read(prev/"registration.json")
    daily, index, raw, taxa, folds, clr, sources = prior.read(PACKAGE, read(prior.DAILY/"config.json"))
    require(all(old["sources"].get(k) == v for k,v in sources.items()), "Archived sources differ")
    pop, periods, early, late = dates.period_means(daily.input_row.to_numpy(int), daily.date.to_numpy(dtype="datetime64[D]"),
                                                  daily[list(dates.COLUMNS)].to_numpy(float))
    groups, family = np.unique(index.iloc[pop].split_unit_id.astype(str), return_inverse=True)
    flow = dict(people=len(pop), households=len(groups), early=int(early.sum()), late=int(late.sum()),
                days=int(early.sum()+late.sum()))
    require(all(flow[k] == old["flow"][k] for k in flow), "Matched cohort changed")
    eco = np.column_stack([prior.raw_ecology(raw[s][pop]) for s in ("oral", "gut")])
    panel_score = panel.build_score(raw["gut"][pop], taxa["gut"])[0]
    scores = np.column_stack((panel_score, eco[:, 2]))
    require(np.isfinite(scores).all(), "A matched participant lacks a marker")
    roles = {key: role for key, role in dates.roles(periods).items() if key in DIRECTIONS}
    def pcs_for(train):
        return prior.site_scores(raw, taxa, index, pop, train, clr, 5)["both_pc"]
    sources.update(old["sources"])
    sources.update({str(path):digest(path) for path in (*ROOT.glob("*.py"), *pins, prev/"COMPLETE.json",
                                                        prev/"registration.json", prev/"summary.json")})
    reg = dict(commit=args.commit, flow=flow, sources=sources, smoke=args.smoke, primary=DIRECTIONS[0],
               seed=SEED, grid=surface.GRID, fixed=surface.FIXED)
    out.mkdir(parents=True, exist_ok=False)
    io.write(out/"registration.json", reg)
    fingerprint = digest(out/"registration.json")
    io.emit("SURFACE_START", smoke=args.smoke, **flow)
    names = ["registration.json", "summary.json", "stage_record.md"]
    for r, f in parts(args.smoke):
        totals, choices = fold_results(roles, eco, scores, family, folds[r, pop]==f, pcs_for, SEED+100*r+f)
        path = out/f"r{r}_f{f}.npz"
        buffer = BytesIO()
        np.savez_compressed(buffer, **totals)
        path.write_bytes(buffer.getvalue())
        io.write(path.with_suffix(".json"), dict(sha=digest(path), registration=fingerprint, choices=choices))
        names.append(path.with_suffix(".json").name)
        io.emit("SURFACE_FOLD", repeat=r, fold=f)
    assoc = {} if args.smoke else direct(roles, eco, pcs_for(np.ones(len(pop), bool)), scores, family, len(groups))
    summary = dict(flow=flow, direct=assoc, prediction=prediction(out, reg), support=dates.support(periods, panel_score))
    io.write(out/"summary.json", summary)
    (out/"stage_record.md").write_text(stage(summary, reg))
    io.write(out/"COMPLETE.json", {n:digest(out/n) for n in names})
    io.emit("SURFACE_COMPLETE", smoke=args.smoke)


if __name__ == "__main__":
    main()
