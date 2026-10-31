"""Evaluation and report of an ADR run: held-out NLL, simple reference models, calibration,
plots and a Markdown report."""
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _softmax_fit(A, labels, K):
    """Multinomial logistic regression (classes 0..K-1, design A with intercept), ridge 1e-3."""
    from scipy.optimize import minimize
    from scipy.special import logsumexp as lse
    p = A.shape[1]

    def f(w):
        W = np.vstack([np.zeros(p), w.reshape(K - 1, p)])
        eta = A @ W.T
        ll = eta[np.arange(len(labels)), labels] - lse(eta, axis=1)
        return -ll.mean() + 1e-3 * (w ** 2).sum()

    w = minimize(f, np.zeros((K - 1) * p), method="L-BFGS-B").x
    W = np.vstack([np.zeros(p), w.reshape(K - 1, p)])
    return lambda B: (B @ W.T) - lse(B @ W.T, axis=1, keepdims=True)


def _discretized_normal_logp(y, m, s, support):
    """log P(Y = y) for a normal law discretized on the support (cells of width step)."""
    h = support.step / 2 if support.values is None else 0.5
    if support.values is not None:  # explicit finite set: normalized normal weights on the set
        v = np.asarray(support.values)
        logw = stats.norm.logpdf(v[None, :], np.atleast_1d(m)[:, None], s)
        logw = logw - np.logaddexp.reduce(logw, axis=1, keepdims=True)
        idx = np.argmin(np.abs(np.asarray(y)[:, None] - v[None, :]), axis=1)
        return logw[np.arange(len(idx)), idx]
    cell = stats.norm.cdf(y + h, m, s) - stats.norm.cdf(y - h, m, s)
    total = stats.norm.cdf(support.upper + h, m, s) - stats.norm.cdf(support.lower - h, m, s)
    return np.log(np.maximum(cell, 1e-300)) - np.log(np.maximum(total, 1e-300))


def _nb_fit(A, y):
    """Negative binomial regression with mean exp(A beta) and constant size, by maximum
    likelihood; returns a function B -> (mean, size)."""
    from scipy.optimize import minimize
    from scipy.special import gammaln

    def nll(th):
        b, logr = th[:-1], th[-1]
        mu, r = np.exp(np.clip(A @ b, -30, 30)), np.exp(logr)
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu))
              + y * np.log(mu / (r + mu)))
        return -ll.mean()

    th0 = np.zeros(A.shape[1] + 1)
    th0[0] = np.log(max(y.mean(), 1e-3))
    th = minimize(nll, th0, method="L-BFGS-B").x
    return lambda B: (np.exp(np.clip(B @ th[:-1], -30, 30)), np.exp(th[-1]))


def _nb_logp(y, mu, r):
    return stats.nbinom.logpmf(y, r, r / (r + mu))


def reference_nll(Xtr, ytr, Xte, yte, mode="continuous", atoms=(), support=None):
    """Held-out NLL of two simple references fitted on the training rows: one without
    covariates and one linear in the covariates. Continuous: normal and linear Gaussian.
    Mixed: atom probabilities (constant, or multinomial logistic in x) and a normal (constant
    or linear Gaussian) for the continuous part. Discrete: a normal law discretized on the
    support (constant mean, or mean linear in x); for counts (support 0, 1, 2, ...), a negative
    binomial law (constant, or with log-linear mean)."""
    out = {}
    A_tr = np.column_stack([np.ones(len(Xtr)), Xtr])
    A_te = np.column_stack([np.ones(len(Xte)), Xte])
    if mode == "mixed":
        a = np.asarray(atoms, dtype=float)
        lab = lambda v: np.array([int(np.argmin(np.abs(a - t))) + 1 if np.min(np.abs(a - t)) <= 1e-9 * max(1, abs(t)) else 0 for t in v])
        ltr, lte = lab(ytr), lab(yte)
        K = len(a) + 1
        cont_tr = ltr == 0
        freq = np.array([(ltr == k).sum() + 0.5 for k in range(K)])
        lp_const = np.log(freq / freq.sum())
        yc = ytr[cont_tr]
        m0, s0 = yc.mean(), max(yc.std(), 1e-6)
        nll0 = -np.where(lte == 0, lp_const[0] + stats.norm.logpdf(yte, m0, s0), lp_const[lte])
        out["atoms + normal, no covariates"] = nll0.mean()
        logp = _softmax_fit(A_tr, ltr, K)(A_te)
        beta, *_ = np.linalg.lstsq(A_tr[cont_tr], yc, rcond=None)
        s1 = max((yc - A_tr[cont_tr] @ beta).std(), 1e-6)
        nll1 = -np.where(lte == 0, logp[:, 0] + stats.norm.logpdf(yte, A_te @ beta, s1),
                         logp[np.arange(len(lte)), lte])
        out["logistic atoms + linear Gaussian"] = nll1.mean()
    elif mode == "discrete" and support.values is None and support.lower == 0 \
            and np.isinf(support.upper) and support.step == 1:
        f0 = _nb_fit(np.ones((len(ytr), 1)), ytr)
        out["negative binomial, no covariates"] = -_nb_logp(yte, *f0(np.ones((len(yte), 1)))).mean()
        f1 = _nb_fit(A_tr, ytr)
        out["negative binomial regression"] = -_nb_logp(yte, *f1(A_te)).mean()
    elif mode == "discrete":
        s0 = max(ytr.std(), support.step if support.values is None else 1e-3)
        out["discretized normal, no covariates"] = -_discretized_normal_logp(yte, ytr.mean(), s0, support).mean()
        beta, *_ = np.linalg.lstsq(A_tr, ytr, rcond=None)
        s1 = max((ytr - A_tr @ beta).std(), support.step if support.values is None else 1e-3)
        out["discretized linear Gaussian"] = -_discretized_normal_logp(yte, A_te @ beta, s1, support).mean()
    elif ytr.ndim == 1:
        out["normal, no covariates"] = -stats.norm(ytr.mean(), ytr.std()).logpdf(yte).mean()
        beta, *_ = np.linalg.lstsq(A_tr, ytr, rcond=None)
        s = (ytr - A_tr @ beta).std()
        out["linear Gaussian"] = -stats.norm(A_te @ beta, s).logpdf(yte).mean()
    else:
        mvn = stats.multivariate_normal(ytr.mean(0), np.cov(ytr.T, bias=True))
        out["normal, no covariates"] = -mvn.logpdf(yte).mean()
        beta, *_ = np.linalg.lstsq(A_tr, ytr, rcond=None)
        res = ytr - A_tr @ beta
        S = np.cov(res.T, bias=True)
        out["linear Gaussian"] = -np.mean([stats.multivariate_normal(m, S).logpdf(v)
                                           for m, v in zip(A_te @ beta, yte)])
    return {k: float(v) for k, v in out.items()}


def evaluate(model, Xte, yte, mode="continuous", atoms=(), support=None, seed=0):
    """Per-row log-likelihood (adr.score.loglik) and, for a univariate response, the PIT values;
    with point masses, the randomized PIT F(y-) + V {F(y) - F(y-)}, V ~ Unif(0, 1)."""
    from adr.score import loglik
    rng = np.random.default_rng(seed)
    lp, pit = [], []
    for x, v in zip(Xte, yte):
        d = model(x)
        v = np.atleast_1d(v)[None] if np.ndim(v) else np.array([v])
        lp.append(float(loglik(d, v, mode, atoms, support)[0]))
        if np.ndim(yte) == 1:
            F = float(np.asarray(d.cdf(v))[0])
            if mode != "continuous":
                with np.errstate(over="ignore"):
                    mass = float(np.exp(np.asarray(d.logmass(v))[0]))
                F = F - rng.uniform() * mass
            pit.append(F)
    return np.array(lp), (np.array(pit) if pit else None)


def plots(run_dir, model, X, y, Xte, yte, pit, cov_names, resp_names, seed=0,
          mode="continuous", atoms=(), support=None):
    """PIT histogram (univariate response), estimated conditional density (one covariate) or
    observed against simulated responses (bivariate response). Returns the file names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    files = []
    if pit is not None:
        fig, ax = plt.subplots(figsize=(4.5, 3.2), layout="constrained")
        ax.hist(pit, bins=20, range=(0, 1), color="#2a78d6", alpha=.75, edgecolor="white")
        ax.axhline(len(pit) / 20, color="black", ls="--", lw=1)
        ax.set(xlabel="PIT = estimated cdf at the observed response (randomized at point masses)", ylabel="count",
               title="Calibration on held-out rows")
        fig.savefig(run_dir / "calibration.png", dpi=150)
        plt.close(fig)
        files.append("calibration.png")
    if y.ndim == 1 and X.shape[1] == 1:
        gx = np.linspace(X[:, 0].min(), X[:, 0].max(), 150)
        if mode == "discrete":
            gy = support.points(y.min(), y.max())
            if len(gy) > 400:
                gy = gy[np.linspace(0, len(gy) - 1, 400).astype(int)]
            vals = np.array([np.exp(np.asarray(model(np.array([g])).logmass(gy))) for g in gx])
            title, label = "Estimated P(Y = y | x)", "probability"
        else:
            pad = 0.1 * (y.max() - y.min())
            gy = np.linspace(y.min() - pad, y.max() + pad, 250)
            vals = np.array([np.exp(np.asarray(model(np.array([g])).logpdf(gy))) for g in gx])
            vals = vals / np.maximum(vals.max(axis=1, keepdims=True), 1e-300)
            title = ("Continuous part of the estimated density" if mode == "mixed"
                     else "Estimated conditional density") + "\n(scaled to 1 at each x)"
        panels = 2 if mode == "mixed" else 1
        fig, axes = plt.subplots(panels, 1, figsize=(6, 4 if panels == 1 else 6), squeeze=False,
                                 sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [3, 1.3][:panels]})
        ax = axes[0, 0]
        ax.pcolormesh(gx, gy, vals.T, cmap="Blues", shading="auto", vmin=0,
                      vmax=None if mode == "discrete" else 1)
        ax.scatter(X[:, 0], y, s=3, color="black", alpha=.35, linewidths=0, label="data")
        ax.set(ylabel=resp_names[0], title=title)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        if mode == "mixed":
            ax2 = axes[1, 0]
            for a in atoms:
                pa = [float(np.exp(np.asarray(model(np.array([g])).logmass(np.array([a])))[0])) for g in gx]
                ax2.plot(gx, pa, lw=1.5, label=f"P(Y = {a:g} | x)")
            ax2.set(ylabel="probability", ylim=(0, 1))
            ax2.legend(frameon=False, fontsize=8)
        axes[-1, 0].set_xlabel(cov_names[0])
        fig.savefig(run_dir / "density.png", dpi=150)
        plt.close(fig)
        files.append("density.png")
    if y.ndim == 2:
        sim = np.array([np.asarray(model(x).sample(rng, 1))[0] for x in Xte])
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, v, t in zip(axes, (yte, sim), ("held-out rows", "one draw per held-out row")):
            ax.scatter(v[:, 0], v[:, 1], s=4, alpha=.5, linewidths=0, color="#2a78d6")
            ax.set(title=t, xlabel=resp_names[0], ylabel=resp_names[1])
        fig.savefig(run_dir / "joint.png", dpi=150)
        plt.close(fig)
        files.append("joint.png")
    return files


def report(run_dir):
    """Write report.md from run.json and the files of the run directory."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())
    s, ev = info["agent"], info.get("evaluation") or {}
    L = [f"# ADR run: `{info['data']}`", "",
         f"Response: {', '.join(f'`{c}`' for c in info['response'])}. "
         f"Covariates ({len(info['covariates'])}): {', '.join(f'`{c}`' for c in info['covariates'])}.",
         f"Training rows: {info['n_train']}; held-out rows: {info['n_test']}.",
         "Response type: " + {"continuous": "continuous (density with respect to Lebesgue measure)",
                              "mixed": "mixed, with point masses at " + ", ".join(f"{a:g}" for a in info.get("atoms") or []),
                              "discrete": f"discrete, on {info.get('support_text')}"}[info.get("mode", "continuous")]
         + ".", "",
         "## Agent run", "",
         *([f"- model: `{s.get('model')}`; time budget: {info['budget_min']} minutes",
            f"- status: {s.get('status')}, after {s.get('minutes')} minutes and {s.get('num_turns')} turns",
            f"- cost reported by Claude Code: {s.get('cost_usd')} USD"] if s.get("model")
           else [f"- {s.get('status')}"]),
         f"- output contract: {'satisfied' if info['contract'] == 'ok' else 'NOT satisfied: ' + '; '.join(info['contract'])}",
         f"- audit flags: {len(info['audit'])}" + ("" if not info["audit"] else
                                                   " (read `task/agent_stream.jsonl` to judge them)")]
    L += [f"  - {a}" for a in info["audit"][:20]]
    if ev:
        L += ["", "## Held-out evaluation", "",
              "Mean negative log-likelihood on the held-out rows (lower is better)"
              + {"continuous": ":", "mixed": "; at the atoms, -log P(Y = y | x), elsewhere -log density:",
                 "discrete": "; -log P(Y = y | x):"}[info.get("mode", "continuous")], "",
              "| model | NLL |", "|---|---|", f"| **ADR** | **{ev['nll']:.4f}** |"]
        L += [f"| {k} | {v:.4f} |" for k, v in ev["reference"].items()]
        if ev.get("coverage"):
            c = ev["coverage"]
            L += ["", f"Coverage of the central 50% and 90% intervals on held-out rows: "
                      f"{c['50']:.1%} and {c['90']:.1%} (nominal 50% and 90%)."]
    for f in info.get("figures", []):
        L += ["", f"![{f}]({f})"]
    L += ["", "## Returned program", "",
          "In `model.py`, `x[j]` is the j-th covariate in this order: "
          + ", ".join(f"`x[{j}]` = `{c}`" for j, c in enumerate(info["covariates"])) + ".",
          "", "```python", (run_dir / "model.py").read_text().rstrip() if (run_dir / "model.py").exists()
          else "# no program", "```"]
    if (run_dir / "notes.md").exists():
        L += ["", "## Notes of the agent", "", (run_dir / "notes.md").read_text().strip()]
    L += ["", "## Using the estimate", "", "```python", "import adr",
          f'fit = adr.load("{run_dir}")', "d = fit.distribution(row)   # row: dict or pandas Series with the covariates",
          "d.logpdf(y); d.cdf(y); d.ppf(0.9); d.sample(rng, 100)", "```", "",
          "Another run on the same data may return a different program. "
          "Method: Tedesco and Finazzi, *Agentic Distributional Regression*, arXiv:2610.37611."]
    (run_dir / "report.md").write_text("\n".join(L) + "\n")
    return run_dir / "report.md"
