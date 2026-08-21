#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import LinearOperator, eigsh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


# =============================================================================
# PATHS / SETTINGS
# =============================================================================
ROOT = Path("/home/ec2-user/Desktop/HPP/Data")
ANALYSIS_DIR = ROOT / "diet_microbiome_analysis"
DATA_DIR = ANALYSIS_DIR / "data"
REPORT_DIR = ANALYSIS_DIR / "reports"
CACHE_DIR = ANALYSIS_DIR / "cache"
FIGURE_DIR = ANALYSIS_DIR / "figures"

for d in [REPORT_DIR, CACHE_DIR, FIGURE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ABUNDANCE_FILE = ROOT / "gut_microbiome_deal/data/05_species_half_min_imputed.csv"
DIET_FILE = DATA_DIR / "00_aligned_four_scores.csv"
IDS_FILE = DATA_DIR / "00_common_participant_ids.csv"

D2_FILE = CACHE_DIR / "02_bray_curtis_squared_float32.dat"
CACHE_META_FILE = CACHE_DIR / "02_bray_curtis_cache_meta.json"

PCOA_FILE = DATA_DIR / "02_pcoa_coordinates.csv"
PERMANOVA_FILE = REPORT_DIR / "02_beta_permanova.csv"
SUMMARY_FILE = REPORT_DIR / "02_beta_diversity_summary.txt"

BLOCK_SIZE = 256
PERMUTATIONS = 999
RANDOM_SEED = 42
PCOA_EIGENVALUES = 4

META_COLS = ["participant_id", "cohort", "research_stage", "array_index"]

SCORES = {
    "AHEI": "AHEI_quintile",
    "AMED": "AMED_quintile",
    "hPDI": "hPDI_quintile",
    "rEDIH": "rEDIH_quintile",
}


# =============================================================================
# HELPERS
# =============================================================================
def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_quintile(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("Q", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    return pd.to_numeric(s, errors="coerce")


def hash_strings(values) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def expected_memmap_bytes(n: int) -> int:
    return n * n * np.dtype(np.float32).itemsize


def cache_is_valid(n: int, participant_ids, species) -> bool:
    if not D2_FILE.exists() or not CACHE_META_FILE.exists():
        return False

    if D2_FILE.stat().st_size != expected_memmap_bytes(n):
        return False

    try:
        meta = json.loads(CACHE_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return False

    expected = {
        "n": n,
        "participant_hash": hash_strings(participant_ids),
        "species_hash": hash_strings(species),
        "species_count": len(species),
    }

    return all(meta.get(k) == v for k, v in expected.items())


def write_cache_meta(n: int, participant_ids, species):
    meta = {
        "n": n,
        "participant_hash": hash_strings(participant_ids),
        "species_hash": hash_strings(species),
        "species_count": len(species),
        "dtype": "float32",
        "metric": "Bray-Curtis squared dissimilarity",
        "source": str(ABUNDANCE_FILE),
    }
    CACHE_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_bray_curtis_squared(X: np.ndarray, participant_ids, species):
    n = X.shape[0]

    if cache_is_valid(n, participant_ids, species):
        print("Bray-Curtis squared-distance cache: VALID -> reusing")
        return np.memmap(D2_FILE, dtype=np.float32, mode="r", shape=(n, n))

    print("Bray-Curtis squared-distance cache: BUILDING")
    print(f"Matrix shape: {n} x {n}")
    print(f"On-disk size: {expected_memmap_bytes(n) / 1024**2:.1f} MiB")

    d2 = np.memmap(D2_FILE, dtype=np.float32, mode="w+", shape=(n, n))

    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_block_pairs = n_blocks * (n_blocks + 1) // 2
    done = 0
    t0 = time.time()

    for bi, i0 in enumerate(range(0, n, BLOCK_SIZE)):
        i1 = min(i0 + BLOCK_SIZE, n)
        Xi = X[i0:i1]

        for j0 in range(i0, n, BLOCK_SIZE):
            j1 = min(j0 + BLOCK_SIZE, n)
            Xj = X[j0:j1]

            dist = cdist(Xi, Xj, metric="braycurtis")
            block_d2 = np.square(dist).astype(np.float32, copy=False)

            d2[i0:i1, j0:j1] = block_d2
            if j0 != i0:
                d2[j0:j1, i0:i1] = block_d2.T

            done += 1
            if done % 25 == 0 or done == total_block_pairs:
                elapsed = time.time() - t0
                print(
                    f"  distance blocks {done}/{total_block_pairs} "
                    f"({100*done/total_block_pairs:.1f}%) | "
                    f"elapsed {elapsed/60:.1f} min",
                    flush=True,
                )

    np.fill_diagonal(d2, 0.0)
    d2.flush()
    write_cache_meta(n, participant_ids, species)

    return np.memmap(D2_FILE, dtype=np.float32, mode="r", shape=(n, n))


def total_inertia_and_row_means(d2: np.ndarray):
    n = d2.shape[0]
    row_sums = np.zeros(n, dtype=np.float64)

    for i0 in range(0, n, BLOCK_SIZE):
        i1 = min(i0 + BLOCK_SIZE, n)
        block = np.asarray(d2[i0:i1, :], dtype=np.float32)
        row_sums[i0:i1] = block.sum(axis=1, dtype=np.float64)

    row_means = row_sums / n
    grand_mean = float(row_means.mean())

    # For G = -1/2 J D^2 J, trace(G) = n * grand_mean / 2.
    total_inertia = 0.5 * n * grand_mean
    return total_inertia, row_means, grand_mean


def compute_pcoa(d2: np.ndarray, participant_ids):
    n = d2.shape[0]
    total_inertia, _, _ = total_inertia_and_row_means(d2)

    print("\n" + "=" * 96)
    print("PCoA")
    print("=" * 96)
    print(f"Total distance-based inertia (trace): {total_inertia:.8f}")

    def matvec(v):
        # Gv = -1/2 * J D^2 J v
        vc = np.asarray(v, dtype=np.float32).copy()
        vc -= vc.mean(dtype=np.float64)

        y = np.empty(n, dtype=np.float32)
        for i0 in range(0, n, BLOCK_SIZE):
            i1 = min(i0 + BLOCK_SIZE, n)
            y[i0:i1] = d2[i0:i1, :] @ vc

        y -= y.mean(dtype=np.float64)
        y *= -0.5
        return y

    operator = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=np.float32,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    v0 = rng.normal(size=n).astype(np.float32)

    t0 = time.time()
    eigenvalues, eigenvectors = eigsh(
        operator,
        k=PCOA_EIGENVALUES,
        which="LA",
        tol=1e-5,
        v0=v0,
        maxiter=500,
    )
    elapsed = time.time() - t0

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = eigenvalues > 0
    if positive.sum() < 2:
        raise RuntimeError("Fewer than two positive PCoA eigenvalues were obtained.")

    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    coordinates = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])[None, :]
    trace_proportion = eigenvalues[:2] / total_inertia

    print(f"PCoA eigensolver elapsed: {elapsed/60:.2f} min")
    print(f"PC1 eigenvalue: {eigenvalues[0]:.8f}")
    print(f"PC2 eigenvalue: {eigenvalues[1]:.8f}")
    print(f"PC1 trace-based proportion: {100*trace_proportion[0]:.3f}%")
    print(f"PC2 trace-based proportion: {100*trace_proportion[1]:.3f}%")
    print(
        "Note: percentages are eigenvalue / total distance-based inertia (trace). "
        "They are useful for comparison but may differ slightly from software that "
        "normalizes using only positive eigenvalues."
    )

    out = pd.DataFrame({
        "participant_id": participant_ids,
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
    })
    out.to_csv(PCOA_FILE, index=False)

    return out, eigenvalues, trace_proportion, total_inertia


# =============================================================================
# PERMANOVA: exact one-factor pseudo-F from squared distance matrix
# =============================================================================
if HAVE_NUMBA:
    @njit(fastmath=True)
    def _observed_within_ss_numba(d2, labels, counts):
        n = labels.shape[0]
        k = counts.shape[0]
        within_pair_sums = np.zeros(k, dtype=np.float64)

        for i in range(n - 1):
            gi = labels[i]
            subtotal = 0.0
            for j in range(i + 1, n):
                if labels[j] == gi:
                    subtotal += d2[i, j]
            within_pair_sums[gi] += subtotal

        within_ss = 0.0
        for g in range(k):
            within_ss += within_pair_sums[g] / counts[g]

        return within_ss


    @njit(parallel=True, fastmath=True)
    def _permutation_stats_numba(d2, permuted_labels, counts, total_ss):
        n_perm, n = permuted_labels.shape
        k = counts.shape[0]
        f_values = np.empty(n_perm, dtype=np.float64)
        r2_values = np.empty(n_perm, dtype=np.float64)

        for p in prange(n_perm):
            labels = permuted_labels[p]
            within_pair_sums = np.zeros(k, dtype=np.float64)

            for i in range(n - 1):
                gi = labels[i]
                subtotal = 0.0
                for j in range(i + 1, n):
                    if labels[j] == gi:
                        subtotal += d2[i, j]
                within_pair_sums[gi] += subtotal

            within_ss = 0.0
            for g in range(k):
                within_ss += within_pair_sums[g] / counts[g]

            between_ss = total_ss - within_ss
            f_values[p] = (
                (between_ss / (k - 1))
                / (within_ss / (n - k))
            )
            r2_values[p] = between_ss / total_ss

        return f_values, r2_values


def observed_permanova_numpy(d2, labels, counts, total_ss):
    k = len(counts)
    within_ss = 0.0

    for g in range(k):
        idx = np.flatnonzero(labels == g)
        sub = d2[np.ix_(idx, idx)]
        within_pair_sum = sub.sum(dtype=np.float64) / 2.0
        within_ss += within_pair_sum / len(idx)

    between_ss = total_ss - within_ss
    f_value = (between_ss / (k - 1)) / (within_ss / (len(labels) - k))
    r2 = between_ss / total_ss
    return within_ss, between_ss, f_value, r2


def run_permanova(d2: np.ndarray, score_name: str, quintiles: pd.Series, total_ss: float):
    q = normalize_quintile(quintiles)
    if q.isna().any():
        raise RuntimeError(f"{score_name}: missing/invalid quintile after alignment.")

    observed_levels = sorted(q.unique().tolist())
    if observed_levels != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"{score_name}: expected Q1-Q5, got {observed_levels}")

    labels = q.astype(int).to_numpy() - 1
    labels = labels.astype(np.int8)
    counts = np.bincount(labels, minlength=5).astype(np.int64)

    # Exact observed statistic.
    if HAVE_NUMBA:
        within_ss = float(_observed_within_ss_numba(d2, labels, counts))
        between_ss = total_ss - within_ss
        f_obs = (between_ss / 4.0) / (within_ss / (len(labels) - 5))
        r2_obs = between_ss / total_ss
    else:
        within_ss, between_ss, f_obs, r2_obs = observed_permanova_numpy(
            d2, labels, counts, total_ss
        )

    rng = np.random.default_rng(RANDOM_SEED)
    permuted_labels = np.empty((PERMUTATIONS, len(labels)), dtype=np.int8)
    for p in range(PERMUTATIONS):
        permuted_labels[p] = rng.permutation(labels)

    t0 = time.time()

    if HAVE_NUMBA:
        f_perm, r2_perm = _permutation_stats_numba(
            d2, permuted_labels, counts, total_ss
        )
    else:
        # Slower fallback. Kept for correctness if numba is unavailable.
        f_perm = np.empty(PERMUTATIONS, dtype=np.float64)
        r2_perm = np.empty(PERMUTATIONS, dtype=np.float64)
        for p in range(PERMUTATIONS):
            _, _, f_perm[p], r2_perm[p] = observed_permanova_numpy(
                d2, permuted_labels[p], counts, total_ss
            )
            if (p + 1) % 50 == 0:
                print(f"  {score_name}: permutations {p+1}/{PERMUTATIONS}", flush=True)

    elapsed = time.time() - t0

    # Standard Monte Carlo permutation P-value; with 999 permutations minimum P=0.001.
    p_value = (1 + np.count_nonzero(f_perm >= f_obs)) / (PERMUTATIONS + 1)

    return {
        "score": score_name,
        "N": len(labels),
        "groups": 5,
        "q1_N": int(counts[0]),
        "q2_N": int(counts[1]),
        "q3_N": int(counts[2]),
        "q4_N": int(counts[3]),
        "q5_N": int(counts[4]),
        "pseudo_F": float(f_obs),
        "R2": float(r2_obs),
        "R2_percent": float(100 * r2_obs),
        "permutations": PERMUTATIONS,
        "p_permutation": float(p_value),
        "permutation_seconds": float(elapsed),
        "perm_F_median": float(np.median(f_perm)),
        "perm_R2_median": float(np.median(r2_perm)),
    }


def save_q1_q5_plots(pcoa: pd.DataFrame, diet: pd.DataFrame, trace_proportion):
    merged = pcoa.merge(
        diet[["participant_id"] + list(SCORES.values())],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    for score, qcol in SCORES.items():
        q = normalize_quintile(merged[qcol])
        mask = q.isin([1, 5])
        plot_df = merged.loc[mask].copy()
        plot_q = q.loc[mask]

        fig, ax = plt.subplots(figsize=(7, 6))

        for level in [1, 5]:
            m = plot_q.eq(level)
            ax.scatter(
                plot_df.loc[m, "PC1"],
                plot_df.loc[m, "PC2"],
                s=10,
                alpha=0.35,
                label=f"{score} Q{level}",
            )

        ax.set_xlabel(f"PC1 ({100*trace_proportion[0]:.2f}% trace inertia)")
        ax.set_ylabel(f"PC2 ({100*trace_proportion[1]:.2f}% trace inertia)")
        ax.set_title(f"Bray-Curtis PCoA: {score} Q1 vs Q5")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"02_pcoa_{score}_Q1_Q5.png", dpi=180)
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 96)
    print("BETA DIVERSITY: BRAY-CURTIS + PCoA + PERMANOVA")
    print("=" * 96)

    ids = pd.read_csv(IDS_FILE, low_memory=False)
    diet = pd.read_csv(DIET_FILE, low_memory=False)
    abundance = pd.read_csv(ABUNDANCE_FILE, low_memory=False)

    for df in [ids, diet, abundance]:
        if "participant_id" not in df.columns:
            raise ValueError("participant_id column missing in an input file.")
        df["participant_id"] = normalize_id(df["participant_id"])

    if ids["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant IDs in common ID file.")
    if abundance["participant_id"].duplicated().any():
        raise RuntimeError("Duplicate participant IDs in abundance file.")

    participant_ids = ids["participant_id"].tolist()

    aligned_abundance = ids[["participant_id"]].merge(
        abundance,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    aligned_diet = ids[["participant_id"]].merge(
        diet,
        on="participant_id",
        how="left",
        validate="one_to_one",
    )

    if aligned_abundance.isna().all(axis=1).any():
        raise RuntimeError("Unexpected completely missing aligned abundance rows.")

    species = [c for c in aligned_abundance.columns if c not in META_COLS]

    print(f"Participants: {len(participant_ids)}")
    print(f"Species: {len(species)}")

    if len(species) != 379:
        raise RuntimeError(f"Expected 379 species, found {len(species)}")

    X = (
        aligned_abundance[species]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(X).all():
        raise RuntimeError("Abundance matrix contains NaN/inf after alignment.")
    if (X < 0).any():
        raise RuntimeError("Negative abundance encountered.")

    row_sums = X.sum(axis=1)
    print("\nAbundance row-sum QC (same imputed scale used for Bray-Curtis):")
    print(pd.Series(row_sums).describe(percentiles=[.01, .50, .99]).to_string())

    d2 = build_bray_curtis_squared(X, participant_ids, species)

    # Exact total SS for distance-based PERMANOVA.
    total_ss = float(d2.sum(dtype=np.float64) / (2.0 * len(participant_ids)))
    print(f"\nDistance total SS: {total_ss:.8f}")

    pcoa, eigenvalues, trace_prop, total_inertia = compute_pcoa(d2, participant_ids)
    save_q1_q5_plots(pcoa, aligned_diet, trace_prop)

    print("\n" + "=" * 96)
    print("PERMANOVA ACROSS DIETARY-SCORE QUINTILES")
    print("=" * 96)
    print(f"Permutations per score: {PERMUTATIONS}")
    print(f"Numba acceleration: {'YES' if HAVE_NUMBA else 'NO'}")

    if HAVE_NUMBA:
        try:
            n_threads = get_num_threads()
            # Avoid excessive oversubscription on very large shared servers.
            if n_threads > 16:
                set_num_threads(16)
            print(f"Numba threads used: {get_num_threads()}")
        except Exception:
            pass

    results = []

    for score, qcol in SCORES.items():
        if qcol not in aligned_diet.columns:
            raise ValueError(f"Missing quintile column for {score}: {qcol}")

        print(f"\n{score}")
        print("-" * 96)

        result = run_permanova(d2, score, aligned_diet[qcol], total_ss)
        results.append(result)

        print(
            f"pseudo-F={result['pseudo_F']:.6f} | "
            f"R2={result['R2_percent']:.4f}% | "
            f"P={result['p_permutation']:.6g} | "
            f"time={result['permutation_seconds']:.1f}s"
        )

    result_df = pd.DataFrame(results)

    def compare_row(row):
        # Paper: all scores P=0.001; overall R2 approximately 0.31%-0.55%.
        if row["p_permutation"] <= 0.01 and 0.1 <= row["R2_percent"] <= 1.0:
            return "MATCH/CLOSE"
        if row["p_permutation"] <= 0.05:
            return "SIGNAL_BUT_DIFFERENT"
        return "NEEDS_INVESTIGATION"

    result_df["paper_comparison"] = result_df.apply(compare_row, axis=1)
    result_df.to_csv(PERMANOVA_FILE, index=False)

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(
        result_df[
            [
                "score", "N", "pseudo_F", "R2_percent",
                "p_permutation", "permutations", "paper_comparison"
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6g}")
    )

    print("\nPAPER REFERENCE")
    print("- Bray-Curtis beta diversity was compared across dietary-score quintiles.")
    print("- PERMANOVA used 999 permutations.")
    print("- All five scores in the paper had P=0.001.")
    print("- Reported PERMANOVA R2 range was approximately 0.31%-0.55%.")
    print("- Q1 vs Q5 showed no clear visual separation in PCoA.")
    print("- Paper PCoA: PC1=11.0%, PC2=7.65% (software normalization may differ).")

    lines = []
    lines.append("BETA DIVERSITY SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Participants: {len(participant_ids)}")
    lines.append(f"Species: {len(species)}")
    lines.append(f"Permutations per score: {PERMUTATIONS}")
    lines.append("")
    lines.append(
        f"PC1 trace-based proportion: {100*trace_prop[0]:.6f}%\n"
        f"PC2 trace-based proportion: {100*trace_prop[1]:.6f}%"
    )
    lines.append("")
    lines.append(
        result_df[
            ["score", "pseudo_F", "R2_percent", "p_permutation", "paper_comparison"]
        ].to_string(index=False)
    )
    lines.append("")
    lines.append("Paper reference: all five dietary patterns P=0.001; R2 0.31%-0.55%.")
    lines.append("Current analysis uses four version-1 dietary scores on the 7,996-person overlap cohort.")
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 96)
    print("SAVED")
    print("=" * 96)
    print(PCOA_FILE)
    print(PERMANOVA_FILE)
    print(SUMMARY_FILE)
    print(FIGURE_DIR)
    print(D2_FILE)


if __name__ == "__main__":
    main()
