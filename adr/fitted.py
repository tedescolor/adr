"""Use the conditional distribution estimated by an ADR run.

    import adr
    fit = adr.load("adr_runs/mydata")        # the output directory of `adr fit`
    print(fit)                               # summary: response, covariates, score, program
    d = fit.distribution({"age": 40})        # adr.dist.Distribution of the response given x
    print(d)                                 # e.g. Normal(loc=1.2, scale=0.3)
    d.logpdf(y), d.cdf(y), d.ppf(q), d.sample(rng, n)
    fit.plot([{"age": 30}, {"age": 60}])     # conditional densities at these covariate values
    fit.logpdf(df)                           # log-likelihood of each row of a data frame
"""
import json
from pathlib import Path

import numpy as np

from adr.contract import load_model


class FittedADR:
    """The estimate of an ADR run: the program ``model.py`` and the information of ``run.json``
    (covariate order, response, response type and evaluation)."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.info = json.loads((self.run_dir / "run.json").read_text())
        self.covariates = self.info["covariates"]
        self.response = self.info["response"]
        self.mode = self.info.get("mode", "continuous")
        self.atoms = self.info.get("atoms") or []
        self.support = None
        if self.info.get("support"):
            from adr.score import Support
            self.support = Support.parse(self.info["support"])
        self.model = load_model(self.run_dir / "model.py")
        self.program = (self.run_dir / "model.py").read_text()

    def _x(self, row):
        """Covariate vector in the column order used for fitting."""
        if hasattr(row, "keys"):
            missing = [c for c in self.covariates if c not in row]
            if missing:
                raise KeyError(f"missing covariates {missing}; needed: {self.covariates}")
            return np.array([float(row[c]) for c in self.covariates])
        x = np.asarray(row, dtype=float).ravel()
        if len(x) != len(self.covariates):
            raise ValueError(f"expected {len(self.covariates)} covariates {self.covariates}")
        return x

    def distribution(self, row):
        """Estimated conditional distribution of the response given one covariate row: a dict or
        pandas Series with the covariate names, or an array in the order of ``covariates``.
        ``print`` it to see the family and its parameters at that row."""
        return self.model(self._x(row))

    def _rows(self, data):
        if hasattr(data, "iterrows"):
            return [r for _, r in data.iterrows()]
        if hasattr(data, "keys"):
            return [data]
        if isinstance(data, (list, tuple)) and data and hasattr(data[0], "keys"):
            return list(data)
        return list(np.atleast_2d(np.asarray(data, dtype=float)))

    def logpdf(self, data, y=None):
        """Log-likelihood of the response at each row (log density for a continuous response,
        log probability at atoms and support points). With a data frame, the response is read
        from its response column(s); otherwise pass y."""
        from adr.score import loglik
        rows = self._rows(data)
        if y is None:
            y = data[self.response].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        if len(self.response) == 1 and y.ndim == 2:
            y = y[:, 0]
        return np.array([float(loglik(self.distribution(r), y[i:i + 1], self.mode, self.atoms,
                                      self.support)[0]) for i, r in enumerate(rows)])

    def quantile(self, row, q):
        """Conditional quantiles at one covariate row (univariate response)."""
        return self.distribution(row).ppf(np.asarray(q, dtype=float))

    def sample(self, data, n=1, seed=None):
        """n draws from the estimated conditional distribution at each row: an array of shape
        (rows, n) for a univariate response and (rows, n, 2) for a bivariate one."""
        rng = np.random.default_rng(seed)
        return np.array([np.asarray(self.distribution(r).sample(rng, n))
                         for r in self._rows(data)])

    def plot(self, rows, ax=None, grid=None):
        """Plot the estimated conditional density (or probabilities) of a univariate response at
        one or several covariate rows; returns the matplotlib axes."""
        import matplotlib.pyplot as plt
        if len(self.response) != 1:
            raise ValueError("plot is available for a univariate response")
        rows = self._rows(rows)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
        rng = np.random.default_rng(0)
        draws = np.concatenate([np.asarray(self.distribution(r).sample(rng, 2000)) for r in rows])
        lo, hi = np.quantile(draws, [0.001, 0.999])
        pad = 0.1 * (hi - lo + 1e-9)
        for k, r in enumerate(rows):
            d = self.distribution(r)
            label = ", ".join(f"{c} = {v:g}" for c, v in zip(self.covariates, self._x(r)))
            color = f"C{k}"
            if self.mode == "discrete":
                pts = self.support.points(lo - pad, hi + pad) if grid is None else np.asarray(grid)
                pm = np.exp(np.asarray(d.logmass(pts)))
                ax.vlines(pts + 0.08 * (k - (len(rows) - 1) / 2) * (pts[1] - pts[0] if len(pts) > 1 else 1),
                          0, pm, color=color, lw=3, label=label)
                ax.set_ylabel("P(Y = y | x)")
            else:
                g = np.linspace(lo - pad, hi + pad, 600) if grid is None else np.asarray(grid)
                masses = [float(np.exp(np.asarray(d.logmass(np.array([a])))[0])) for a in self.atoms]
                if masses:
                    label += "; " + ", ".join(f"P(Y = {a:g}) = {m:.2f}" for a, m in zip(self.atoms, masses))
                ax.plot(g, np.exp(np.asarray(d.logpdf(g))), color=color, lw=1.6, label=label)
                for a in self.atoms:
                    ax.plot([a], [0], marker="v", color=color, ms=8)
                ax.set_ylabel("conditional density" + (" (continuous part)" if self.atoms else ""))
        ax.set_xlabel(self.response[0])
        ax.legend(frameon=False, fontsize=8)
        return ax

    def __repr__(self):
        i = self.info
        kind = {"continuous": "continuous",
                "mixed": "mixed, point masses at " + ", ".join(f"{a:g}" for a in self.atoms),
                "discrete": f"discrete on {i.get('support_text')}"}[self.mode]
        nc = len(self.covariates)
        lines = [f"ADR estimate of {', '.join(self.response)} given {nc} covariate{'s' * (nc != 1)} "
                 f"({i.get('data')})",
                 "  covariates: " + ", ".join(f"x[{j}] = {c}" for j, c in enumerate(self.covariates)),
                 f"  response type: {kind}"]
        ev = i.get("evaluation")
        if ev:
            lines.append(f"  held-out NLL: {ev['nll']:.4f} (" + "; ".join(
                f"{k}: {v:.4f}" for k, v in ev["reference"].items()) + ")")
        lines += ["  program (model.py):", ""] + ["    " + l for l in self.program.rstrip().splitlines()]
        return "\n".join(lines)


def load(run_dir):
    """Load the estimate written by `adr fit` in its output directory (the --out folder)."""
    return FittedADR(run_dir)
