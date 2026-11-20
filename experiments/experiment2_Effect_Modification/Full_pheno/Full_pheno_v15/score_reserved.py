"""Score the three prespecified assays on previously frozen V15 predictions."""

from pathlib import Path
import sys

from freeze_reserved import V5, select_values
from run_repeat import ENDPOINTS, HERE, MARGIN, emit, parse, read_json, require, v1_run, write_json


HEADS = ["A", "reference:B", "ridge_100:B", "reference:fixed", "reference:half",
         "ridge_100:fixed", "ridge_100:half"]


def paired(y, predicted, households):
    import numpy as np
    from scipy.stats import t

    people = len(y)
    _, codes, counts = np.unique(households, return_inverse=True, return_counts=True)
    families = len(counts)
    sst = float(((y - y.mean()) ** 2).sum()) if people else 0.
    if families < 40 or sst <= 1e-10 * max(np.finfo(float).tiny, float(y @ y)):
        return dict(scorable=False, people=people, households=families, p_one_sided=1.)
    errors = (y[:, None] - predicted) ** 2
    sse = errors.sum(axis=0)
    multiplier = 100. / sst
    gain = errors[:, 2] - errors[:, 6]
    diff = gain.sum()
    centred = np.bincount(codes, weights=gain) - counts * diff / people
    se = np.sqrt(families / (families - 1.) * (centred @ centred))
    probability = float(t.sf(diff / se, families - 1)) if se > 0. else 1.
    return dict(scorable=True, people=people, households=families, p_one_sided=probability,
                main_pp=float(multiplier * (sse[0] - sse[2])),
                interaction_pp=float(multiplier * diff),
                total_pp=float(multiplier * (sse[0] - sse[6])),
                reference_half_total_pp=float(multiplier * (sse[0] - sse[4])),
                matched_improvement_pp=float(multiplier * (sse[4] - sse[6])),
                r2_by_arm={head: float(1. - loss / sst) for head, loss in zip(HEADS, sse, strict=True)})


def main():
    args = parse(__doc__)
    v1_run.init_runtime(1)
    sys.path.insert(1, str(V5))
    import numpy as np
    import pyarrow.parquet as pq
    from reserved_inputs import KEY, _arrow_frame
    from run_expanded_grid import holm

    out = HERE / "runs/TRE_ONLY" / args.run_id
    require((out / "FROZEN.json").is_file(), "V15_NO_FROZEN_PREDICTIONS")
    with v1_run.run_lock(out):
        frozen = read_json(out / "FROZEN.json")
        reg = read_json(out / "registration.json")
        pred = out / "predictions.npz"
        require(frozen["heads"] == HEADS and reg["endpoints"] == list(ENDPOINTS)
                and frozen["confirmation_outcomes_read"] is False, "V15_FROZEN_SCOPE")
        require(v1_run.file_sha(pred) == frozen["predictions_sha"]
                and v1_run.file_sha(out / "registration.json") == frozen["registration_sha"]
                and all(v1_run.file_sha(Path(p)) == h for p, h in reg["code"].items()), "V15_FROZEN_INPUT_CHANGED")
        role_file = HERE.parents[2] / "inputs/discovery_v2/roles_v2_full_TRE_ONLY.parquet"
        require(v1_run.file_sha(role_file) == reg["predictors"][str(role_file)], "V15_FROZEN_ROLE_CHANGED")
        roster = _arrow_frame(pq.ParquetFile(role_file).read(
            columns=KEY + ["split_unit_id", "final_role"], use_threads=False))
        with np.load(pred, allow_pickle=False) as data:
            ids, houses = data["participant_id"], data["households"]
            lo, hi = data["lower"], data["upper"]
            values = [data[f"assay{j}"] for j in range(len(ENDPOINTS))]
        index = roster.set_index("participant_id").loc[ids].reset_index()
        require(len(ids) == len(set(ids)) == len(index) == frozen["participants"]
                and index.final_role.eq("confirmation_reserved").all()
                and np.array_equal(index.split_unit_id.to_numpy(dtype=str), houses)
                and not index.split_unit_id.isin(roster.loc[roster.final_role.eq("discovery"), "split_unit_id"]).any(),
                "V15_RESERVED_SCORING_ROSTER")
        require(all(x.shape == (len(ids), len(HEADS)) and np.isfinite(x).all() for x in values)
                and np.isfinite(lo).all() and np.isfinite(hi).all() and (lo <= hi).all(),
                "V15_INVALID_FROZEN_PAYLOAD")
        index["input_row"] = np.arange(len(index))
        sources = {k: Path(record[0]) for k, record in reg["outcome_source"].items()}
        require(all(v1_run.file_sha(sources[k]) == rec[1] for k, rec in reg["outcome_source"].items()),
                "V15_OUTCOME_REGISTERED_SOURCE_CHANGED")
        receipt = dict(commit=args.commit, score_code_sha=v1_run.file_sha(Path(__file__)),
                       frozen_sha=v1_run.file_sha(out / "FROZEN.json"), outcomes=list(ENDPOINTS),
                       sources=reg["outcome_source"], access="registered reserved prediction keys only")
        if (out / "score_access.json").exists():
            require(read_json(out / "score_access.json") == receipt, "V15_SCORE_ACCESS_MISMATCH")
        else:
            write_json(out / "score_access.json", receipt)
        y = select_values(sources, index, ENDPOINTS)
        records = []
        for j, endpoint in enumerate(ENDPOINTS):
            keep = np.isfinite(y[:, j])
            target = np.clip(y[keep, j], lo[j], hi[j])
            records.append(dict(endpoint=endpoint, **paired(target, values[j][keep], houses[keep])))
        holm(records)
        for row in records:
            row["met_joint_threshold"] = bool(row["scorable"] and row["p_holm"] < .05
                                              and row["interaction_pp"] >= MARGIN and row["total_pp"] > 0.)
        result = dict(commit=args.commit, reserved_outcomes_read=True, records=records)
        write_json(out / "score_summary.json", result)
    emit("V15_RESERVED_SCORED", result)


if __name__ == "__main__":
    main()
