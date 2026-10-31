"""Response types, supports and the score (log-likelihood) of a program.

Three response types are supported. For each, the log-likelihood is taken with respect to a
reference measure fixed before the fit, the same for all the models compared:

- ``continuous``: Lebesgue measure; the score is the log density (the setting of the paper);
- ``mixed``: Lebesgue measure plus a unit mass at each declared atom; at an atom ``a`` the
  score is log P(Y = a | x), elsewhere the log density of the continuous part;
- ``discrete``: counting measure on a declared support (finite or infinite); the score is
  log P(Y = y | x).

Supports (``Support.parse``): ``naturals`` (0, 1, 2, ...), ``integers``, ``a:b`` (the integers
from a to b), ``a:b:s`` (a, a+s, ..., b), ``a:inf`` or ``a:inf:s`` (no upper bound),
``-inf:b`` (no lower bound), or an explicit list ``v1,v2,...``.
"""
import math

import numpy as np

TOL = 1e-9


class Support:
    def __init__(self, lower=-math.inf, upper=math.inf, step=1.0, values=None):
        self.lower, self.upper, self.step = float(lower), float(upper), float(step)
        self.values = None if values is None else sorted(float(v) for v in values)

    @classmethod
    def parse(cls, spec):
        spec = str(spec).strip()
        if spec == "naturals":
            return cls(0, math.inf, 1)
        if spec == "integers":
            return cls(-math.inf, math.inf, 1)
        if ":" in spec:
            parts = spec.split(":")
            lo, hi = float(parts[0]), float(parts[1]) if parts[1] else math.inf
            step = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
            if math.isinf(lo) and math.isinf(hi):
                return cls(-math.inf, math.inf, step)
            return cls(lo, hi, step)
        return cls(values=[float(v) for v in spec.split(",") if v.strip()])

    @property
    def finite(self):
        return self.values is not None or not (math.isinf(self.lower) or math.isinf(self.upper))

    @property
    def offset(self):
        return self.lower if not math.isinf(self.lower) else (self.upper if not math.isinf(self.upper) else 0.0)

    def contains(self, y):
        y = np.asarray(y, dtype=float)
        if self.values is not None:
            v = np.asarray(self.values)
            return np.any(np.abs(y[..., None] - v) <= TOL * np.maximum(1, np.abs(v)), axis=-1)
        k = (y - self.offset) / self.step
        on = np.abs(k - np.round(k)) <= 1e-7
        return on & (y >= self.lower - TOL) & (y <= self.upper + TOL)

    def points(self, lo, hi):
        """The support points in [lo, hi] (for plots and references)."""
        if self.values is not None:
            return np.array([v for v in self.values if lo <= v <= hi])
        a = max(lo, self.lower)
        b = min(hi, self.upper)
        k0 = math.ceil((a - self.offset) / self.step - 1e-9)
        k1 = math.floor((b - self.offset) / self.step + 1e-9)
        return self.offset + self.step * np.arange(k0, k1 + 1)

    def spec(self):
        if self.values is not None:
            return ",".join(f"{v:g}" for v in self.values)
        if self.lower == 0 and math.isinf(self.upper) and self.step == 1:
            return "naturals"
        if math.isinf(self.lower) and math.isinf(self.upper) and self.step == 1:
            return "integers"
        f = lambda v: "inf" if v == math.inf else ("-inf" if v == -math.inf else f"{v:g}")
        return f"{f(self.lower)}:{f(self.upper)}" + ("" if self.step == 1 else f":{self.step:g}")

    def describe(self):
        if self.values is not None:
            return "the values " + ", ".join(f"{v:g}" for v in self.values)
        if self.step == 1 and self.lower == 0 and math.isinf(self.upper):
            return "the non-negative integers 0, 1, 2, ... (no upper bound)"
        if self.step == 1 and math.isinf(self.lower) and math.isinf(self.upper):
            return "the integers (no lower or upper bound)"
        lo = "no lower bound" if math.isinf(self.lower) else f"from {self.lower:g}"
        hi = "no upper bound" if math.isinf(self.upper) else f"to {self.upper:g}"
        return f"the values on the grid with step {self.step:g}, {lo}, {hi}"


def loglik(d, y, mode="continuous", atoms=(), support=None):
    """Log-likelihood of the observations y under the distribution d, for the response type."""
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if mode == "continuous":
        return np.asarray(d.logpdf(y), dtype=float)
    if mode == "mixed":
        a = np.asarray(atoms, dtype=float)
        at = np.any(np.abs(y[:, None] - a[None, :]) <= TOL * np.maximum(1, np.abs(a)), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(at, d.logmass(y), d.logpdf(y)).astype(float)
    if mode == "discrete":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(support.contains(y), d.logmass(y), -np.inf).astype(float)
    raise ValueError(f"unknown response type {mode}")


# ------------------------------------------------------------------ inspection of a response
def lattice_step(values, max_den=20):
    """The step g of a grid containing all the (distinct) values, or None."""
    v = np.unique(np.asarray(values, dtype=float))
    if len(v) < 2:
        return None
    d = np.diff(v)
    base = d.min()
    for den in range(1, max_den + 1):
        g = base / den
        k = (v - v[0]) / g
        if np.all(np.abs(k - np.round(k)) <= 1e-6 * np.maximum(1, np.abs(k))):
            return float(np.round(g, 12))
    return None


def detect_atoms(y, share=0.01, min_count=5):
    """Values that occur at least max(min_count, share * n) times."""
    y = np.asarray(y, dtype=float)
    vals, counts = np.unique(y, return_counts=True)
    thr = max(min_count, share * len(y))
    return [float(v) for v, c in zip(vals, counts) if c >= thr]


def analyse_response(y):
    """Summary of a univariate response and the suggested response type."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    vals, counts = np.unique(y, return_counts=True)
    tied = counts[counts > 1].sum() / n
    atoms = detect_atoms(y)
    step = lattice_step(vals) if len(vals) <= max(50, n // 2) else None
    if step is not None and (vals.max() - vals.min()) / step > 2 * n:
        step = None  # a grid much finer than the data is only the recording precision
    order = np.argsort(-counts)[:8]
    out = {"n": n, "distinct": int(len(vals)), "share_tied": float(tied),
           "min": float(y.min()), "max": float(y.max()),
           "most_frequent": [(float(vals[i]), int(counts[i])) for i in order if counts[i] > 1],
           "candidate_atoms": atoms, "grid_step": step}
    non_atom = ~np.isin(y, atoms)
    tied_rest = 0.0
    if non_atom.any():
        _, c2 = np.unique(y[non_atom], return_counts=True)
        tied_rest = c2[c2 > 1].sum() / non_atom.sum()
    notes = []
    if step is not None and tied >= 0.5:
        out["type"] = "discrete"
        integer = abs(step - round(step)) < 1e-9 and np.all(np.abs(vals - np.round(vals)) < 1e-9)
        if integer and step == 1 and y.min() >= 0:
            sup = Support(0, math.inf, 1)
        elif integer and step == 1:
            sup = Support(-math.inf, math.inf, 1)
        elif y.min() >= 0 and Support(0, math.inf, step).contains(y).all():
            sup = Support(0, math.inf, step)
        elif Support(-math.inf, math.inf, step).contains(y).all():
            sup = Support(-math.inf, math.inf, step)
        else:
            sup = Support(float(y.min()), math.inf, step)
        out["support"] = sup.spec()
        notes.append(f"The observed values lie between {y.min():g} and {y.max():g}. The suggested "
                     f"support ({sup.describe()}) has no upper bound. If the response cannot "
                     f"exceed a maximum (a score, a number of successes out of m trials, ...), "
                     f"give a finite support, for example --support {y.min():g}:{y.max():g}"
                     + ("" if step == 1 else f":{step:g}") + ".")
        if len(vals) <= 10:
            notes.append(f"Only {len(vals)} distinct values: the support may well be finite.")
        if len(vals) > 30 and step < 1:
            notes.append(f"If the response is a continuous measurement rounded to {step:g}, it can "
                         "also be fitted as continuous.")
    elif atoms and tied_rest < 0.2:
        out["type"] = "mixed"
        out["atoms"] = atoms
        share = np.isin(y, atoms).mean()
        notes.append(f"{share:.1%} of the responses are exactly equal to "
                     + ", ".join(f"{a:g}" for a in atoms) + "; the other values are almost all distinct.")
    elif tied >= 0.2:
        out["type"] = "continuous"
        notes.append(f"{tied:.0%} of the responses share their value with another row: the data may "
                     "be rounded or discrete. Consider --discrete with an explicit --support, or "
                     "check how the response was recorded.")
    else:
        out["type"] = "continuous"
    out["notes"] = notes
    return out
