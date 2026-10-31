"""Distributions returned by agent-written programs.

Every object returned by ``model(x)`` must be an instance of ``Distribution``.
Univariate families are thin wrappers around ``scipy.stats`` frozen distributions and
accept array-valued parameters (one value per point), so that they can be used as
conditional factors inside ``Sequential``.

Multivariate responses are built from
- ``MultivariateNormal(mean, cov)``;
- ``Independent(d1, d2, ...)``: independent univariate coordinates;
- ``Sequential(first, cond)``: p(y1) * p(y2 | y1), where ``cond(y1)`` returns the
  distribution of y2 given y1 (y1 is passed as an array; use array-valued parameters);
- ``Mixture`` of any of the above (all components must have the same dimension).

Shapes: a univariate distribution has ``logpdf(y)`` for y of shape (n,) and ``sample``
returning (n,); a d-variate one uses (n, d).
"""
import numpy as np
from scipy import stats
from scipy.special import expit, logsumexp

SIGMA_MIN = 1e-6  # lower bound on every scale parameter


class Distribution:
    def logpdf(self, y):
        raise NotImplementedError

    def cdf(self, y):
        raise NotImplementedError

    def sample(self, rng, n=1):
        raise NotImplementedError

    def ppf(self, q):
        """Quantile function (univariate only); default: numerical inversion of the cdf."""
        q = np.asarray(q, dtype=float)
        lo, hi = self._ppf_bracket(q)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < q
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out = 0.5 * (lo + hi)
        out = np.where(q <= 0, -np.inf, out)
        return np.where(q >= 1, np.inf, out)

    def _ppf_bracket(self, q):
        s = self.sample(np.random.default_rng(0), 2000)
        span = s.max() - s.min() + 1.0
        return (np.full_like(q, s.min() - 10 * span), np.full_like(q, s.max() + 10 * span))


class Scipy(Distribution):
    def __init__(self, frozen):
        self.d = frozen

    def logpdf(self, y):
        return self.d.logpdf(y)

    def cdf(self, y):
        return self.d.cdf(y)

    def ppf(self, q):
        return self.d.ppf(q)

    def sample(self, rng, n=1):
        return self.d.rvs(size=n, random_state=rng)


def _arr(v):
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite distribution parameter")
    return v


def _scale(s):
    return np.maximum(_arr(s), SIGMA_MIN)


def Normal(loc, scale):
    return Scipy(stats.norm(_arr(loc), _scale(scale)))


def StudentT(df, loc, scale):
    return Scipy(stats.t(np.maximum(_arr(df), 0.1), _arr(loc), _scale(scale)))


def SkewNormal(loc, scale, shape):
    return Scipy(stats.skewnorm(_arr(shape), _arr(loc), _scale(scale)))


def Laplace(loc, scale):
    return Scipy(stats.laplace(_arr(loc), _scale(scale)))


def LogNormal(mu, sigma):
    return Scipy(stats.lognorm(_scale(sigma), scale=np.exp(_arr(mu))))


def Gamma(shape, rate):
    return Scipy(stats.gamma(_scale(shape), scale=1.0 / _scale(rate)))


class MultivariateNormal(Distribution):
    def __init__(self, mean, cov):
        self.mean = _arr(mean).ravel()
        cov = _arr(cov)
        d = len(self.mean)
        if cov.shape != (d, d):
            raise ValueError("cov must be a d x d matrix")
        cov = 0.5 * (cov + cov.T) + SIGMA_MIN**2 * np.eye(d)
        self.d = stats.multivariate_normal(self.mean, cov)

    def logpdf(self, y):
        return np.atleast_1d(self.d.logpdf(np.asarray(y).reshape(-1, len(self.mean))))

    def sample(self, rng, n=1):
        return np.asarray(self.d.rvs(size=n, random_state=rng)).reshape(n, len(self.mean))


def Cov2(sd1, sd2, rho):
    """2x2 covariance matrix from two standard deviations and a correlation."""
    sd1, sd2 = float(_scale(sd1)), float(_scale(sd2))
    rho = float(np.clip(_arr(rho), -0.999, 0.999))
    return [[sd1 * sd1, rho * sd1 * sd2], [rho * sd1 * sd2, sd2 * sd2]]


class Independent(Distribution):
    def __init__(self, *components):
        if not all(isinstance(c, Distribution) for c in components):
            raise TypeError("Independent components must be Distributions")
        self.components = components

    def logpdf(self, y):
        y = np.asarray(y).reshape(-1, len(self.components))
        return sum(c.logpdf(y[:, j]) for j, c in enumerate(self.components))

    def sample(self, rng, n=1):
        return np.column_stack([c.sample(rng, n) for c in self.components])


class Sequential(Distribution):
    """Bivariate p(y1, y2) = p(y1) p(y2 | y1); ``cond(y1)`` maps an array of y1 values to
    the (vectorised) univariate distribution of y2."""

    def __init__(self, first, cond):
        if not isinstance(first, Distribution) or not callable(cond):
            raise TypeError("Sequential(first: Distribution, cond: callable)")
        self.first, self.cond = first, cond

    def logpdf(self, y):
        y = np.asarray(y, dtype=float).reshape(-1, 2)
        return self.first.logpdf(y[:, 0]) + self.cond(y[:, 0]).logpdf(y[:, 1])

    def sample(self, rng, n=1):
        y1 = np.asarray(self.first.sample(rng, n)).reshape(n)
        y2 = np.asarray(self.cond(y1).sample(rng, n)).reshape(n)
        return np.column_stack([y1, y2])


class Mixture(Distribution):
    MAX_COMPONENTS = 5

    def __init__(self, weights, components):
        if len(weights) != len(components) or not 1 <= len(weights) <= self.MAX_COMPONENTS:
            raise ValueError(f"Mixture needs 1..{self.MAX_COMPONENTS} components, one weight each")
        w = np.stack(np.broadcast_arrays(*[_arr(v) for v in weights]))
        if np.any(w < 0) or np.any(w.sum(axis=0) <= 0):
            raise ValueError("mixture weights must be non-negative")
        if not all(isinstance(c, Distribution) for c in components):
            raise TypeError("mixture components must be Distributions")
        self.w = w / w.sum(axis=0)
        self.components = list(components)

    def logpdf(self, y):
        with np.errstate(divide="ignore"):
            lw = np.log(self.w)
        terms = [lw[k] + c.logpdf(y) for k, c in enumerate(self.components)]
        return logsumexp(np.stack(np.broadcast_arrays(*terms)), axis=0)

    def cdf(self, y):
        return sum(self.w[k] * c.cdf(y) for k, c in enumerate(self.components))

    def sample(self, rng, n=1):
        w = self.w.reshape(len(self.components), -1)
        k = (rng.random(n)[None, :] > np.cumsum(w, axis=0)).sum(axis=0)
        k = np.minimum(k, len(self.components) - 1)
        draws = np.stack([np.asarray(c.sample(rng, n)) for c in self.components])
        return draws[k, np.arange(n)]

    def _ppf_bracket(self, q):
        # a mixture quantile lies between its components' quantiles
        qs = np.clip(q, 1e-12, 1 - 1e-12)
        comp = np.stack([c.ppf(qs) for c in self.components])
        return comp.min(axis=0), comp.max(axis=0)


# elementary functions available to programs
sin, cos, exp, log, sqrt, tanh, arctan2 = np.sin, np.cos, np.exp, np.log, np.sqrt, np.tanh, np.arctan2
abs, maximum, minimum, where = np.abs, np.maximum, np.minimum, np.where
pi = np.pi


def sigmoid(z):
    return expit(z)


def softmax(*z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=0))
    return list(e / e.sum(axis=0))


def dot(a, b):
    return float(np.dot(a, b))


__all__ = [
    "Distribution", "Normal", "StudentT", "SkewNormal", "Laplace", "LogNormal", "Gamma",
    "MultivariateNormal", "Cov2", "Independent", "Sequential", "Mixture",
    "sin", "cos", "exp", "log", "sqrt", "tanh", "arctan2", "abs", "maximum", "minimum", "where",
    "pi", "sigmoid", "softmax", "dot",
]
