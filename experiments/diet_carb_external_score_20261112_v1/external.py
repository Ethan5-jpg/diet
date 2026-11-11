"""Within-person and between-person tests of a literature-anchored gut score."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.stats import t

SOURCE = Path(__file__).resolve().parent
BASE_NAME = "diet_daily_modification_20261112_v1"


def verified_base(folder, expected):
    for name, sha in expected.items():
        if sha256((folder / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f"Pinned daily analysis source changed: {name}")
    sys.path.insert(0, str(folder))
    try:
        import inputs
        import models
        import run
    finally:
        sys.path.pop(0)
    if any(Path(module.__file__).resolve().parent != folder.resolve() for module in (inputs, models, run)):
        raise ImportError("Loaded unexpected analysis dependency")
    return inputs, models, run


def panel_scores(abundance, names, mapping):
    labels = [name.rsplit("|", 1)[-1] for name in names]
    mapped = {original: label for original, label in mapping.items() if label is not None}
    if len(set(mapped.values())) != len(mapped) or any(labels.count(label) != 1 for label in mapped.values()):
        raise ValueError("Every mapped literature taxon needs one distinct source column")
    if not (np.isfinite(abundance).all() and (abundance >= 0).all()
            and np.allclose(abundance.sum(axis=1), 1, atol=1e-3, rtol=0)):
        raise ValueError("Gut profiles must be finite nonnegative relative abundances")
    idx = [labels.index(label) for label in mapped.values()]
    total = abundance[:, idx].sum(axis=1)
    coli = abundance[:, labels.index(mapped["Escherichia coli"])]
    return {"sum7": total, "log_sum7": np.log1p(100 * total),
            "sum_without_coli": np.maximum(0, total - coli)}


def person_means(daily):
    return daily.groupby("person_index", sort=True).agg(
        carb=("carb", "mean"), fiber=("fiber", "mean"), protein=("protein", "mean"),
        energy=("energy_kcal", "mean"), y=("Y_value", "mean"),
        age=("age_calendar_years", "first"), sex=("sex_code", "first"),
        households=("split_unit_id", "first"), n_days=("Y_value", "size"),
    )


def between_design(table, score, models):
    carb, fiber = (models.scaled_dose(d, table[d]) for d in ("carb", "fiber"))
    protein = (table.protein.to_numpy(float) - 15) / 10
    age = (table.age.to_numpy(float) - 50) / 10
    sex = table.sex.to_numpy(float)
    b = np.column_stack([np.ones(len(score)), carb, fiber, protein, table.energy / 1000,
                         carb ** 2, fiber ** 2, protein ** 2, carb * fiber, carb * protein,
                         age, sex, table.n_days, score, score ** 2,
                         *(d * c for d in (carb, fiber, protein) for c in (age, sex))])
    return table.y.to_numpy(float), b, np.ones(len(score)), np.column_stack([carb * score, fiber * score])


def candidates():
    for label, columns in (("carb", [0]), ("fiber", [1]), ("joint", [0, 1])):
        yield label, columns


def report_test(z, fitted, clusters, absorbed, cols, score, models):
    beta, cov, p = models.household_test(z[:, cols], fitted, clusters, absorbed)
    step = np.quantile(score, 0.75) - np.quantile(score, 0.25)
    scale = t.ppf(0.975, len(set(clusters)) - 1)
    values = {}
    for j, col in enumerate(cols):
        estimate = float(beta[j] * step)
        margin = float(scale * step * np.sqrt(cov[j, j]))
        values[models.DIETS[col]] = dict(q75_minus_q25=estimate,
                                        ci95=[estimate - margin, estimate + margin])
    return dict(p=p, contrasts=values)


def evaluate(daily, per_person, scores, designs, models, penalty, train_people=None):
    people = daily.person_index.to_numpy()
    use_holdout = train_people is not None
    if train_people is None:
        train_people = np.ones(len(per_person), bool)
    output = {}
    for representation, values in scores.items():
        score = models.train_standardize(values[:, None], train_people)[:, 0]
        within = (*designs, models.design(daily, models.DIETS, score[people, None], False)[0])
        between = between_design(per_person, score, models)
        for label, data, row_map, clusters, absorbed in (
            ("within", within, people, daily.split_unit_id.astype(str).to_numpy(), len(score)),
            ("between", between, np.arange(len(score)), per_person.households.astype(str).to_numpy(), 0),
        ):
            y, x, w, z = data
            tr = train_people[row_map]
            base = models.fit_background(x[tr], y[tr], w[tr])
            for term, cols in candidates():
                key = f"{label}_{representation}_{term}"
                if use_holdout:
                    _, sse0, sse = models.fit_interaction(x, z[:, cols], y, w, tr, ~tr, base, penalty)
                    reference = y[tr].mean() if label == "between" else 0.0
                    tss = float(np.sum(w[~tr] * (y[~tr] - reference) ** 2))
                    output[key] = dict(sse=sse, sse_null=sse0, tss=tss)
                else:
                    output[key] = report_test(z, base, clusters, absorbed, cols, score, models)
    if not use_holdout:
        for row, adjusted in zip(output.values(), models.holm([row["p"] for row in output.values()])):
            row["holm_p"] = float(adjusted)
    return output


def aggregate(folds):
    summary = {}
    for repeat in sorted({fold["repeat"] for fold in folds}):
        rows = [fold["results"] for fold in folds if fold["repeat"] == repeat]
        for key in rows[0]:
            tss, sse0, sse = (sum(row[key][name] for row in rows) for name in ("tss", "sse_null", "sse"))
            summary.setdefault(key, {})[f"r{repeat}"] = dict(
                null_r2=1 - sse0 / tss, r2=1 - sse / tss, delta_r2=(sse0 - sse) / tss
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    if ((package / "experiments" / SOURCE.name).resolve() != SOURCE
            or not re.fullmatch(r"[0-9a-f]{40}", args.commit)
            or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run)):
        raise ValueError("Invalid package path, commit or run ID")
    config = json.loads((SOURCE / "config.json").read_bytes())
    base_path = SOURCE.parent / BASE_NAME
    inputs, models, io = verified_base(base_path, config["base_code"])
    base_cfg = inputs.json_read(base_path / "config.json")
    code = {name: inputs.digest(SOURCE / name) for name in ("external.py", "config.json")}
    daily, _, raw, taxa, partitions, _, data_hashes = inputs.read(package, base_cfg)
    population = daily.input_row.drop_duplicates().to_numpy(dtype=int)
    daily["person_index"] = daily.input_row.map({row: j for j, row in enumerate(population)})
    per_person = person_means(daily)
    scores = panel_scores(raw["gut"][population], taxa["gut"], config["taxa"])
    designs = models.background(daily)
    flow = dict(people=len(population), days=len(daily), households=int(per_person.households.nunique()),
                mapped_taxa=sum(v is not None for v in config["taxa"].values()),
                panel_zeros=int((scores["sum7"] == 0).sum()))
    io.emit("EXTERNAL_START", commit=args.commit, **flow)
    registration = dict(commit=args.commit, sources=data_hashes, code=code, base=config["base_code"],
                        smoke=args.smoke, flow=flow, python=sys.version.split()[0],
                        versions={name: version(name) for name in ("numpy", "pandas", "pyarrow", "scipy")})
    out = SOURCE / "runs_TRE_ONLY" / args.run
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = out / ".running"
    lock.mkdir()
    try:
        (lock / "pid").write_text(str(os.getpid()), encoding="ascii")
        reg = out / "registration.json"
        if args.resume:
            inputs.require(inputs.json_read(reg) == registration and not (out / "COMPLETE.json").exists(),
                           "Resume requires an unfinished run with matching registration")
        else:
            io.write(reg, registration)
        sha = inputs.digest(reg)
        schedule = [(0, 0)] if args.smoke else [(r, f) for r in (0, 1) for f in range(5)]
        checkpoints = []
        for repeat, fold in schedule:
            saved = out / f"r{repeat}_f{fold}.json"
            if saved.is_file():
                row = inputs.json_read(saved)
                inputs.require((row["repeat"], row["fold"], row["registration"]) == (repeat, fold, sha),
                               "Saved fold provenance does not match")
            else:
                train = partitions[repeat, population] != fold
                row = dict(repeat=repeat, fold=fold, registration=sha, results=evaluate(
                    daily, per_person, scores, designs, models, config["penalty"], train
                ))
                io.write(saved, row)
                io.emit("EXTERNAL_FOLD_DONE", repeat=repeat, fold=fold, validation_people=int((~train).sum()))
            checkpoints.append(row)
        summary = dict(registration=sha, folds=len(checkpoints), flow=flow, prediction=aggregate(checkpoints))
        if not args.smoke:
            summary["direct"] = evaluate(daily, per_person, scores, designs, models, config["penalty"])
            for key in ("within_sum7_carb", "between_sum7_carb"):
                io.emit("EXTERNAL_CONTRAST", specification=key, **summary["direct"][key],
                        prediction=summary["prediction"][key])
        inputs.require(all(inputs.digest(Path(p)) == v for p, v in data_hashes.items())
                       and all(inputs.digest(SOURCE / p) == v for p, v in code.items())
                       and all(inputs.digest(base_path / p) == v for p, v in config["base_code"].items()),
                       "Sources changed during external score analysis")
        io.write(out / "summary.json", summary)
        io.write(out / "COMPLETE.json", dict(summary=inputs.digest(out / "summary.json"), registration=sha))
        io.emit("EXTERNAL_COMPLETE", run=args.run, smoke=args.smoke, folds=len(checkpoints))
    finally:
        (lock / "pid").unlink(missing_ok=True)
        lock.rmdir()


if __name__ == "__main__":
    main()
