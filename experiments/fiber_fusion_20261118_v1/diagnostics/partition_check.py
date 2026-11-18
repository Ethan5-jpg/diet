"""Freeze each discovery winner, replay one archive fold, then score on two unused family partitions."""
import os
for env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env] = "1"

import argparse
from io import BytesIO
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fusion

SEEDS = (2026111891, 2026111892)
LABELS = ("off", "anchor", "winner", "matched_off", "matched_anchor", "matched_winner")


def select(summary):
    return {k:max(fusion.METHODS, key=lambda n:v[n]["r2"]) for k,v in summary.items()}


def score(by_repeat):
    n = len(LABELS)
    r2, mae = [], []
    for a in by_repeat:
        total = a.sum(axis=0)
        r2.append(1-total[2:2+n]/total[1])
        mae.append(total[2+n:]/total[0])
    r2, mae = np.asarray(r2), np.asarray(mae)
    result = {}
    for name, j in (("original", 0), ("matched", 3)):
        off, ref, on = r2[:, j:j+3].T
        gains = on-ref
        contrib = np.stack([(a[:, 3+j]-a[:, 4+j])/a[:, 1].sum() for a in by_repeat])
        np.testing.assert_allclose(contrib.sum(axis=1), gains, atol=1e-12)
        keep = np.ones(contrib.shape[1], bool)
        keep[np.argsort(-abs(contrib.mean(axis=0)), kind="stable")[:int(np.ceil(.01*len(keep)))]] = False
        trimmed = [(a[keep,3+j]-a[keep,4+j]).sum()/a[keep,1].sum() for a in by_repeat]
        result[name] = dict(anchor_r2=float(ref.mean()), winner_r2=float(on.mean()), delta_by_repeat=gains.tolist(),
                            trimmed_delta=float(np.mean(trimmed)), mae_gain=float(np.mean(mae[:, j+1]-mae[:, j+2])),
                            on_pct=float(np.mean(100*(on-off)/on)) if np.all(on>0) else None,
                            off_pct=float(np.mean(100*(on-off)/off)) if np.all(off>0) else None)
    return result


def predict(m, inputs, omit, balances, held, picks):
    flow, roles, ctx, eco, scores, fam, _, pcs_for = inputs
    train = ~held
    m.require(not set(fam[train]) & set(fam[held]), "Household overlap")
    pcs = pcs_for(train)
    payload = {}
    for i, (k, other) in enumerate(zip(m.s.DIRS, reversed(m.s.DIRS))):
        target = roles[k]
        full, _ = m.s.prev.ext.predictions(m.prev.surface, m.old.pattern, target, roles[other], eco, pcs, scores, ctx[k], train, held)
        anchor = full[:, (3, 5)[i]].copy()
        if i:
            anchor += m.prev.comp.correction(m.prev.surface, target, target, eco, pcs, scores, omit, train, held)[1]
        additive, delta = fusion.additions(m.prev.surface, target, eco, pcs, scores, balances, omit, ctx[k], train, held, bool(i))
        pred = np.column_stack((full[:, 0], anchor, anchor+delta[:, fusion.METHODS.index(picks[k])]))
        payload[k] = m.old.pattern.loss_totals(target[0], np.column_stack((pred, pred+additive[:, None])), train, held, fam, flow["households"])
    return payload


def collect(m, out, reg):
    rows = {}
    for tag in reg["parts"]:
        p = out/(tag+".npz")
        m.require(m.read(p.with_suffix(".json")) == dict(sha=m.digest(p), registration=m.digest(out/"registration.json")), "Bad fold")
        with np.load(p, allow_pickle=False) as data:
            for k in m.s.DIRS:
                key = (k, tag.split("_")[0])
                a = data[k]
                m.require(a.shape == (reg["flow"]["households"], 14), "Bad household schema")
                rows[key] = rows.get(key, 0)+a
    if reg["smoke"]:
        return {"replay": "passed"}
    counts = rows[m.s.DIRS[0], "s0"][:,0]
    m.require(min(counts)>0 and counts.sum()==reg["flow"]["people"] and all(np.array_equal(a[:,0],counts) for a in rows.values()), "Unmatched families")
    return {k:score([rows[k, "s0"], rows[k, "s1"]]) for k in m.s.DIRS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run")
    p.add_argument("--commit")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        a = np.full((120, 14), 5.)
        a[:,0:2] = [1., 10.]
        a[:,2:8] = [8., 7., 6., 8., 7., 6.]
        fit = score([a, a])["original"]
        np.testing.assert_allclose([*fit["delta_by_repeat"], fit["trimmed_delta"], fit["on_pct"], fit["off_pct"]], [.1, .1, .1, 50, 100])
        assert select({"k":{n:{"r2":i} for i,n in enumerate(fusion.METHODS)}})["k"] == fusion.METHODS[-1]
        print("PARTITION_SYNTHETIC_PASS")
        return
    import run as m
    m.require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.run or "") and re.fullmatch(r"[0-9a-f]{40}", args.commit or ""), "Bad run or release")
    archive = ROOT/"runs_TRE_ONLY/discovery-1"
    out = ROOT/"diagnostics_TRE_ONLY"/args.run
    parent = m.read(archive/"registration.json")
    m.require(parent["commit"] == "fcbbfb414c00367e51b308cc2603c97ef4874dc0" and m.old.receipt(archive) == m.read(archive/"COMPLETE.json"), "Discovery receipt differs")
    picks = select(m.read(archive/"summary.json"))
    source = dict(parent["sources"])
    source.update({str(p):m.digest(p) for p in (archive/"COMPLETE.json", archive/"summary.json", Path(__file__))})
    m.require(all(m.digest(Path(p)) == v for p,v in source.items()), "Source mismatch")
    smoke = m.read(out/"registration.json")["smoke"] if args.verify else args.smoke
    tags = ["replay_0"] if smoke else [f"s{i}_{f}" for i in range(2) for f in range(5)]
    reg = dict(commit=args.commit, picks=picks, smoke=smoke, parts=tags, flow=parent["flow"], sources=source, seeds=list(SEEDS))
    if args.verify:
        m.require(reg == m.read(out/"registration.json") and collect(m, out, reg) == m.read(out/"summary.json"), "Readback mismatch")
        m.require(m.read(out/"COMPLETE.json") == {n:m.digest(out/n) for n in ("registration.json", "summary.json", "stage_record.md")}, "Incomplete")
        print("PARTITION_READBACK_PASS", len(tags), flush=True)
        return
    m.require(not out.exists(), "Use a fresh run directory")
    total, omit, balances, fam = m.markers(parent)
    inputs = m.old.inputs(parent)
    m.require(np.array_equal(fam, inputs[5]) and np.array_equal(total, inputs[4][:,0]), "Alignment mismatch")
    previous = m.s.load_folds(fam, inputs[6])
    fresh = [m.s.st.partition(fam, seed) for seed in SEEDS]
    def signature(arr):
        return tuple(sorted(tuple(np.flatnonzero(arr==f)) for f in range(5)))
    sig = [signature(arr) for arr in (*previous, *fresh)]
    m.require(len(set(sig)) == len(sig), "Reused a partition")
    out.mkdir(parents=True)
    m.s.io.write(out/"registration.json", reg)
    for tag in tags:
        split, f = tag.split("_")
        held = (previous[2] if smoke else fresh[int(split[1:])]) == int(f)
        data = predict(m, inputs, omit, balances, held, picks)
        if smoke:
            with np.load(archive/"r2_f0.npz", allow_pickle=False) as ref:
                for k, n in picks.items():
                    cols = [fusion.NAMES.index(v) for v in ("off", "anchor", n, "matched_off", "matched_anchor", "matched_"+n)]
                    cols = [0,1]+[2+j for j in cols]+[2+len(fusion.NAMES)+j for j in cols]
                    np.testing.assert_allclose(data[k], ref[k][:,cols], rtol=2e-8, atol=2e-7)
        path = out/(tag+".npz")
        buf = BytesIO()
        np.savez_compressed(buf, **data)
        path.write_bytes(buf.getvalue())
        m.s.io.write(path.with_suffix(".json"), dict(sha=m.digest(path), registration=m.digest(out/"registration.json")))
        print("PARTITION_FOLD", tag, flush=True)
    result = collect(m, out, reg)
    m.s.io.write(out/"summary.json", result)
    lines = ["# Fixed winner on new discovery partitions", "", f"Frozen choices {picks}.", f"Partition seeds {SEEDS}.",
             "Original and matched backgrounds are scored separately. Trimmed delta removes the 1% most influential households by absolute mean paired contribution.",
             "These partitions reuse the discovery participants. A new partition measures partition sensitivity."]
    if not smoke:
        for k, rows in result.items():
            for bg,v in rows.items():
                lines.append(f"{k} {bg} {picks[k]} | R2 {v['winner_r2']:.7f} | gains {v['delta_by_repeat']} | trimmed {v['trimmed_delta']:+.7f} | MAE gain {v['mae_gain']:+.6f} | on % {v['on_pct']} | off % {v['off_pct']}")
    (out/"stage_record.md").write_text("\n".join(lines)+"\n")
    m.s.io.write(out/"COMPLETE.json", {n:m.digest(out/n) for n in ("registration.json", "summary.json", "stage_record.md")})
    print("PARTITION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
