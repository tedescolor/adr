"""Use the conditional distribution estimated by an ADR run.

    import adr
    fit = adr.load("adr_runs/mydata")        # the --out directory of `adr fit`
    d = fit.distribution(df.iloc[0])         # adr.dist.Distribution for the response
    d.logpdf(y), d.cdf(y), d.ppf(q), d.sample(rng, n)
    fit.logpdf(df)                           # log-density of each row of a data frame
    fit.sample(df, n=100, seed=0)            # draws for each row
"""
import json
from pathlib import Path

import numpy as np

from adr.contract import load_model


class FittedADR:
    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        info = json.loads((run_dir / "run.json").read_text())
        self.covariates = info["covariates"]
        self.response = info["response"]
        self.model = load_model(run_dir / "model.py")
        self.program = (run_dir / "model.py").read_text()

    def _x(self, row):
        """Covariate vector in the column order used for fitting."""
        if hasattr(row, "keys"):
            return np.array([float(row[c]) for c in self.covariates])
        x = np.asarray(row, dtype=float).ravel()
        if len(x) != len(self.covariates):
            raise ValueError(f"expected {len(self.covariates)} covariates {self.covariates}")
        return x

    def distribution(self, row):
        """Estimated conditional distribution of the response given one covariate row
        (a dict or pandas Series with the covariate names, or an array in fitting order)."""
        return self.model(self._x(row))

    def _rows(self, data):
        if hasattr(data, "iterrows"):
            return [r for _, r in data.iterrows()]
        return list(np.atleast_2d(np.asarray(data, dtype=float)))

    def logpdf(self, data, y=None):
        """Log-density of the response at each row. With a data frame, the response is read
        from its response column(s); otherwise pass y."""
        rows = self._rows(data)
        if y is None:
            y = data[self.response].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        if len(self.response) == 1 and y.ndim == 2:
            y = y[:, 0]
        out = [float(np.asarray(self.distribution(r).logpdf(y[i:i + 1]))[0])
               for i, r in enumerate(rows)]
        return np.array(out)

    def sample(self, data, n=1, seed=None):
        """n draws from the estimated conditional distribution at each row: an array of shape
        (rows, n) for a univariate response and (rows, n, 2) for a bivariate one."""
        rng = np.random.default_rng(seed)
        return np.array([np.asarray(self.distribution(r).sample(rng, n))
                         for r in self._rows(data)])

    def __repr__(self):
        return (f"FittedADR(response={self.response}, covariates={self.covariates})\n"
                + self.program)


def load(run_dir):
    """Load the result of `adr fit` (its --out directory)."""
    return FittedADR(run_dir)
