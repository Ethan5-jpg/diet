"""Training-only representations; no selection by held-out outcomes."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class FixedMemberTarget:
    """Reuse training robust limits, keeping the previously fixed eligibility."""
    def __init__(self, original):
        self.original = original

    def fit(self, y, participant_ids, *, endpoint, role='training'):
        self.original.fit(y, participant_ids, endpoint=endpoint, role=role)
        self.state = dict(self.original.state)
        self.state['original_kind'] = self.state.get('kind')
        self.state['kind'] = 'functional_fixed_membership_training_clip_only_v1'
        self.state['new_8sd_exclusions_applied'] = False
        return self

    def transform(self, y, *, endpoint):
        result = self.original.transform(y, endpoint=endpoint)
        values = np.asarray(y, float)
        finite = np.isfinite(values)
        cleaned = np.full(values.shape, np.nan)
        cleaned[finite] = np.clip(values[finite], self.state['clip_low'], self.state['clip_high'])
        result['values'] = cleaned
        result['original_rule_excluded_8sd'] = result.get('excluded_8sd')
        result['excluded_8sd'] = np.zeros(values.shape, bool)
        result['winsorized_low'] = finite & (values < self.state['clip_low'])
        result['winsorized_high'] = finite & (values > self.state['clip_high'])
        return result


class FunctionState:
    def __init__(self, mode, config):
        self.mode, self.config = mode, config

    def fit(self, x):
        x = np.asarray(x, float)
        self.keep = np.flatnonzero(np.mean(np.isfinite(x) & (x > 0), axis=0) >= self.config['minimum_training_signal_prevalence'])
        self.observed = []
        if self.mode == 'abundance_unknown_null':
            self.observed = [np.sort(c[np.isfinite(c)]) for c in x[:, self.keep].T]
            ranks = self._ranks(x)
            self.rank_medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else .5 for c in ranks.T])
        raw = self.raw(x)
        self.input_scaler = StandardScaler(with_std=self.mode == 'abundance_unknown_null').fit(raw) if raw.shape[1] else None
        z = self.input_scaler.transform(raw) if self.input_scaler is not None else raw
        self.pca, self.pc_scaler = None, None
        self.pc_keep = np.array([], int)
        count = min(self.config['functional_main_components'], z.shape[1], max(len(z)-1, 0))
        if count and np.any(z.std(0) > 1e-10):
            self.pca = PCA(n_components=count, svd_solver='full').fit(z)
            scores = self.pca.transform(z)
            self.pc_keep = np.flatnonzero(scores.std(0) > 1e-10)
            if len(self.pc_keep):
                self.pc_scaler = StandardScaler().fit(scores[:, self.pc_keep])
        return self

    def _ranks(self, x):
        result = np.full((len(x), len(self.keep)), np.nan)
        for j, (column, observed) in enumerate(zip(self.keep, self.observed)):
            present = np.isfinite(x[:, column])
            values = x[present, column]
            if len(observed):
                result[present, j] = (np.searchsorted(observed, values, 'left') + np.searchsorted(observed, values, 'right')) / (2.0*len(observed))
        return result

    def raw(self, x):
        x = np.asarray(x, float)
        if np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0):
            raise ValueError('Invalid nonnegative functional signals')
        if self.mode == 'abundance_relative':
            values = np.where(np.isfinite(x), x, 0.)
            sums = values.sum(1)
            if (sums <= 0).any():
                raise ValueError('Entire missing/zero profile must remain flagged')
            return np.sqrt(values[:, self.keep]/sums[:, None])
        if self.mode == 'coverage_confidence':
            if np.any(x[np.isfinite(x)] > 1+1e-8):
                raise ValueError('Supplied coverage score outside observed range')
            return np.sqrt(np.where(np.isfinite(x[:, self.keep]), x[:, self.keep], 0.))
        if self.mode != 'abundance_unknown_null':
            raise ValueError('Unknown functional route')
        ranks = self._ranks(x)
        return np.column_stack([np.where(np.isfinite(ranks), ranks, self.rank_medians), ~np.isfinite(ranks)]).astype(float)

    def transform(self, x):
        raw = self.raw(np.asarray(x, float))
        if self.pca is None or self.pc_scaler is None:
            return np.empty((len(raw), 0))
        z = self.input_scaler.transform(raw)
        return self.pc_scaler.transform(self.pca.transform(z)[:, self.pc_keep])

    def audit(self, names):
        result = dict(mode=self.mode, retained_fields=len(self.keep), pc_dimensions=len(self.pc_keep))
        if self.pca is not None:
            result['explained_variance_ratio'] = self.pca.explained_variance_ratio_.tolist()
            labels = [names[j] for j in self.keep]
            if self.mode == 'abundance_unknown_null':
                labels = ['observed_rank:'+n for n in labels] + ['reporting_mask:'+n for n in labels]
            result['unsupervised_top_loadings'] = []
            for pc in self.pca.components_[self.pc_keep[:self.config['functional_modifier_components']]]:
                selected = np.argsort(-abs(pc), kind='stable')[:8]
                result['unsupervised_top_loadings'].append([dict(field=labels[j], loading=float(pc[j])) for j in selected])
        return result


class TechnicalState:
    def raw(self, x):
        a = np.asarray(x, float)
        good = np.isfinite(a) & (a >= 0)
        result = np.full(a.shape, np.nan)
        result[good] = np.log1p(a[good])
        return result

    def fit(self, x):
        a = self.raw(x)
        self.median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in a.T])
        self.scaler = StandardScaler().fit(self.complete(a))
        return self

    def complete(self, a):
        return np.column_stack([np.where(np.isfinite(a), a, self.median), ~np.isfinite(a)]).astype(float)

    def transform(self, x):
        return self.scaler.transform(self.complete(self.raw(x)))


class Design:
    def __init__(self, base, mode, signals, technical, config, frozen_features):
        self.base, self.role, self.config = base, base.role, base.config
        self.functional_config, self.frozen_features = config, frozen_features
        self.mode = mode
        s = self.role.split
        self.function = FunctionState(mode, config).fit(signals[s.train_rows])
        self.pcs = [self.function.transform(signals[rows]) for rows in (s.train_rows, s.rows)]
        number = min(config['functional_curved_components'], self.pcs[0].shape[1])
        self.function_curves = frozen_features.Curves().fit(self.pcs[0][:,:number]) if number else None
        self.curved_components = number
        self.technical_state = TechnicalState().fit(technical[s.train_rows])
        self.technical = [self.technical_state.transform(technical[rows]) for rows in (s.train_rows, s.rows)]
        self.extra_scaler = StandardScaler().fit(self.extra(True))
        self.xt, self.xv = self.main(True), self.main(False)
        self.interaction_states = {}

    def extra(self, training, value=None):
        pos = 0 if training else 1
        diet, tech = self.base.diet_basis(training, value), self.technical[pos]
        curves = self.function_curves.transform(self.pcs[pos][:,:self.curved_components]) if self.function_curves is not None else np.empty((len(tech),0))
        return np.column_stack([self.pcs[pos], curves, tech, tech*diet[:, :1], tech*diet[:, 1:2]])

    def main(self, training=False, value=None):
        return np.column_stack([self.base.main(training, value), self.extra_scaler.transform(self.extra(training, value))])

    def interaction_raw(self, spec, training, value=None):
        pos = 0 if training else 1
        number = self.functional_config['functional_modifier_components']
        return self.pcs[pos][:, :number]*self.base.diet_basis(training, value)[:, :1]

    def interactions(self, spec):
        if self.pcs[0].shape[1] == 0:
            state = self.frozen_features.InteractionMap(np.empty((self.xt.shape[1], 0)), np.array([], int), None)
            return np.empty((len(self.xt), 0)), np.empty((len(self.xv), 0)), state
        return self.frozen_features.Design.interactions(self, spec)

    def support(self):
        pieces = []
        for pos in (0,1):
            background = self.base.background_t if pos == 0 else self.base.background_v
            pieces.append(np.column_stack([background, self.base.pcs['oral'][pos][:, :6], self.base.pcs['gut'][pos][:, :6], self.pcs[pos][:, :3], self.technical[pos]]))
        scale = StandardScaler().fit(pieces[0])
        n = min(self.functional_config['support_neighbors'], len(pieces[0]))
        neighbors = NearestNeighbors(n_neighbors=n).fit(scale.transform(pieces[0]))
        rows = neighbors.kneighbors(scale.transform(pieces[1]), return_distance=False)
        limits = np.quantile(self.base.diet(True)[rows], self.functional_config['support_quantiles'], axis=1)
        lo, hi = self.config['contrast']
        return (limits[0] <= lo) & (limits[1] >= hi)
