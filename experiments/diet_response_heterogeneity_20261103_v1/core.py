"""Profiled ridge background with ridge/elastic-net interaction blocks.

Only NumPy/SciPy are required. All scales and projections use training rows.
The elastic-net objective retains the *same* ridge main-effect penalty;
it does not fit ordinary elastic net to an unpenalized residual regression.
"""
import io
from pathlib import Path
import numpy as np
from scipy import linalg, stats


def rcs(x, knots):
    a, b, c = np.asarray(knots, float)
    if not a < b < c:
        raise ValueError("Three distinct spline knots required")
    x = np.asarray(x, float)
    return (np.maximum(x-a, 0)**3 - (c-a)/(c-b)*np.maximum(x-b, 0)**3
            + (b-a)/(c-b)*np.maximum(x-c, 0)**3) / (c-a)**2


def save_npz(path, **arrays):
    """Fresh sequential write: no seek/append on a TRE mounted filesystem."""
    path = Path(path)
    if any(np.asarray(x).dtype.hasobject for x in arrays.values()):
        raise ValueError("Object arrays are prohibited")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())
    with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as check:
        for name, expected in arrays.items():
            np.testing.assert_array_equal(check[name], expected)


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite evaluation values")
    mse = float(np.mean((y-prediction)**2))
    variance = float(np.var(y))
    return dict(n=len(y), MSE=mse, RMSE=float(np.sqrt(mse)),
                R2=1-mse/variance if variance else None)


def cluster_fit(y, x, groups):
    """Unweighted OLS with household CR1 and t(G-1) intervals, diagnostic only."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    n, p = x.shape
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite clustered regression")
    if np.linalg.matrix_rank(x) != p:
        raise ValueError("Rank-deficient clustered regression")
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    G = int(inverse.max())+1
    if G <= 1 or n <= p:
        raise ValueError("Insufficient cluster degrees of freedom")
    bread = np.linalg.inv(x.T@x)
    beta = bread@(x.T@y)
    residual = y-x@beta
    sums = np.zeros((G, p))
    np.add.at(sums, inverse, x*residual[:, None])
    covariance = bread@(sums.T@sums)@bread * G/(G-1) * (n-1)/(n-p)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    critical = stats.t.ppf(.975, G-1)
    return beta, covariance, np.column_stack([beta-critical*se, beta+critical*se])


class Background:
    """min .5 mean SSE + .5 main_lambda ||standardized CLR coefficients||²."""
    def __init__(self, base, z, y, training, main_lambda=1.):
        self.base, self.z, self.y = map(lambda a: np.asarray(a, float), (base, z, y))
        self.training = np.asarray(training, bool)
        self.n = int(self.training.sum())
        self.main_lambda = float(main_lambda)
        if self.n < 3 or self.main_lambda <= 0:
            raise ValueError("Nonempty training and positive main penalty required")
        b, zz, yy = self.base[self.training], self.z[self.training], self.y[self.training]
        if not all(np.isfinite(v).all() for v in (b, zz, yy)):
            raise ValueError("Nonfinite training background")
        _, r, piv = linalg.qr(b, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        rank = int(np.sum(diagonal > max(diagonal.max(initial=0), 1)*1e-10))
        self.keep = np.sort(piv[:rank])
        self.b = b[:, self.keep]
        self.z_scale = zz.std(0)
        self.z_scale[self.z_scale <= 1e-12] = 1.
        zz = zz/self.z_scale
        self.z_projection = np.linalg.lstsq(self.b, zz, rcond=None)[0]
        self.y_projection = np.linalg.lstsq(self.b, yy, rcond=None)[0]
        self.z_residual = zz-self.b@self.z_projection
        self.y_residual = yy-self.b@self.y_projection
        gram = self.z_residual.T@self.z_residual/self.n
        gram.flat[::len(gram)+1] += self.main_lambda
        self.factor = linalg.cho_factor(gram, lower=True, check_finite=False)
        self.z_rhs = self.z_residual.T@self.y_residual/self.n
        self.null_z = linalg.cho_solve(self.factor, self.z_rhs, check_finite=False)

    def interaction(self, raw):
        return Interaction(self, raw)

    def null(self):
        return self._recover(self.null_z, None, None)

    def _recover(self, z_coefficient, interaction, coefficient):
        bc = self.y_projection-self.z_projection@z_coefficient
        if interaction is not None:
            bc = bc-interaction.projection@coefficient
        full_bc = np.zeros(self.base.shape[1])
        full_bc[self.keep] = bc
        raw_z = z_coefficient/self.z_scale
        prediction = self.base@full_bc+self.z@raw_z
        raw_w = np.zeros(0) if interaction is None else coefficient/interaction.scale
        if interaction is not None:
            prediction = prediction+interaction.raw@raw_w
        return dict(base_coef=full_bc, main_coef=raw_z, interaction_coef=raw_w,
                    prediction=prediction, base_keep=self.keep.copy())


class Interaction:
    def __init__(self, background, raw):
        self.background = background
        self.raw = np.asarray(raw, float)
        train = self.raw[background.training]
        if train.ndim != 2 or not train.shape[1] or not np.isfinite(train).all():
            raise ValueError("Invalid training interaction block")
        self.scale = train.std(0)
        self.scale[self.scale <= 1e-12] = 1.
        train = train/self.scale
        self.projection = np.linalg.lstsq(background.b, train, rcond=None)[0]
        residual = train-background.b@self.projection
        self.cross = background.z_residual.T@residual/background.n
        self.z_solution = linalg.cho_solve(background.factor, self.cross, check_finite=False)
        gram = residual.T@residual/background.n-self.cross.T@self.z_solution
        self.gram = (gram+gram.T)/2
        self.rhs = residual.T@background.y_residual/background.n-self.cross.T@background.null_z

    def finish(self, coef, **diagnostics):
        b = self.background
        fit = b._recover(b.null_z-self.z_solution@coef, self, coef)
        fit.update(diagnostics)
        return fit

    def ridge(self, penalty):
        if penalty <= 0:
            raise ValueError("Positive interaction penalty required")
        matrix = self.gram.copy()
        matrix.flat[::len(matrix)+1] += penalty
        coef = linalg.cho_solve(linalg.cho_factor(matrix, lower=True), self.rhs)
        return self.finish(coef, penalty=float(penalty), kkt_error=0., iterations=1)

    def elastic_net(self, alpha, fraction, initial=None, tolerance=1e-7, max_iter=5000):
        if not 0 < alpha <= 1 or not 0 < fraction <= 1:
            raise ValueError("alpha and lambda fraction must be in (0, 1]")
        lam = float(np.max(np.abs(self.rhs), initial=0)/alpha*fraction)
        l1, l2 = alpha*lam, (1-alpha)*lam
        coef = np.zeros(len(self.rhs)) if initial is None else np.asarray(initial, float).copy()
        if lam == 0 or fraction == 1.:
            return self.finish(np.zeros_like(coef), penalty=lam, alpha=alpha,
                               lambda_fraction=fraction, kkt_error=0., iterations=0,
                               standardized_coef=np.zeros_like(coef))
        diagonal = np.diag(self.gram)+l2
        if np.any(diagonal <= 0):
            raise ValueError("Nonpositive coordinate curvature")
        residual = self.rhs-self.gram@coef
        threshold = tolerance*max(1., float(np.max(np.abs(self.rhs))))
        for iteration in range(1, max_iter+1):
            for j in range(len(coef)):
                value = residual[j]+self.gram[j, j]*coef[j]
                updated = np.sign(value)*max(abs(value)-l1, 0)/diagonal[j]
                change = updated-coef[j]
                if change:
                    residual -= change*self.gram[:, j]
                    coef[j] = updated
            gradient = self.gram@coef-self.rhs+l2*coef
            active = coef != 0
            violation = np.where(active, np.abs(gradient+l1*np.sign(coef)),
                                 np.maximum(np.abs(gradient)-l1, 0))
            error = float(violation.max(initial=0))
            if error <= threshold:
                return self.finish(coef, penalty=lam, alpha=float(alpha),
                                   lambda_fraction=float(fraction), kkt_error=error,
                                   iterations=iteration, standardized_coef=coef.copy())
        raise RuntimeError(f"Elastic net did not converge: KKT {error} > {threshold}")
