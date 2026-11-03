import numpy as np
import pandas as pd
K=['participant_id', 'cohort', 'research_stage']
KD=K+["date"]
def window(d):
    version = str(d.mapping_version.iloc[0]) if len(d) else 'empty'
    rows = []
    for (key, z) in d.groupby(K, sort=False):
        a = dict(zip(K, key))
        a.update(window_id='food_window', n_recorded_days=len(z), first_recorded_date=z.date.min(), last_recorded_date=z.date.max(), mapping_version=version)
        e = z.loc[z.energy_eligible]
        a['n_energy_days'] = len(e)
        a['n_unknown_NOVA_days'] = int((e.unknown_NOVA_kcal > 0).sum())
        total = float(e.energy_kcal.sum())
        k = float(e.known_UPF_kcal.sum())
        u = float(e.unknown_NOVA_kcal.sum())
        a.update(energy_kcal_sum_energy_days=total if len(e) else np.nan, UPF_known_kcal_sum_energy_days=k if len(e) else np.nan, NOVA_unknown_kcal_sum_energy_days=u if len(e) else np.nan, NOVA_coverage_energyweighted=1 - u / total if total > 0 else np.nan)
        for (label, col) in [('UPF_zeroU', 'UPF_point_zero_unknown_energy_pp'), ('UPF_strict', 'UPF_point_all_rows_classified_pp'), ('fiber', 'fiber_density_g_per_1000kcal')]:
            t = z.loc[z[col].notna()]
            a['n_' + label + '_days'] = len(t)
            a[label + '_equal_day_mean'] = float(t[col].mean()) if len(t) else np.nan
            a[label + '_numerator_sum'] = float(t.fiber_g.sum() if label == 'fiber' else t.known_UPF_kcal.sum()) if len(t) else np.nan
            a[label + '_energy_denominator_sum'] = float(t.energy_kcal.sum()) if len(t) else np.nan
            a[label + '_cumulative_ratio'] = (1000 if label == 'fiber' else 100) * a[label + '_numerator_sum'] / a[label + '_energy_denominator_sum'] if len(t) else np.nan
            a[label + '_first_valid_date'] = t.date.min() if len(t) else None
            a[label + '_last_valid_date'] = t.date.max() if len(t) else None
            a[label + '_flag_ge7_DIAGNOSTIC'] = len(t) >= 7
        a['UPF_lower_equal_day_mean_all_energy_days'] = float(e.UPF_lower_pp.mean()) if len(e) else np.nan
        a['UPF_upper_equal_day_mean_all_energy_days'] = float(e.UPF_upper_pp.mean()) if len(e) else np.nan
        a['UPF_lower_cumulative_all_energy_days'] = 100 * k / total if total > 0 else np.nan
        a['UPF_upper_cumulative_all_energy_days'] = 100 * (k + u) / total if total > 0 else np.nan
        a['UPF_exact_full_energy_window_equal_day_mean'] = a['UPF_lower_equal_day_mean_all_energy_days'] if len(e) > 0 and u == 0 else np.nan
        a['UPF_exact_full_energy_window_cumulative_ratio'] = 100 * k / total if total > 0 and u == 0 else np.nan
        rows.append(a)
    return pd.DataFrame(rows)

def common_view(d, cgm):
    required = KD + ['cgm_day_eligible', 'point_qc_verified']
    assert set(required) <= set(cgm)
    assert not cgm.duplicated(KD).any(), 'CGM daily keys must be unique'
    assert cgm.point_qc_verified.eq(True).all(), 'W09 verification required'
    out = d.merge(cgm.loc[cgm.cgm_day_eligible, required], on=KD, how='inner', validate='one_to_one')
    return out
