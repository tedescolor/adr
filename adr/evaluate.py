"""Evaluation and report of an ADR run: held-out NLL, simple reference models, calibration,
plots and a Markdown report."""
import json
from pathlib import Path

import numpy as np
from scipy import stats


def reference_nll(Xtr, ytr, Xte, yte):
    """Held-out NLL of two simple Gaussian references fitted by maximum likelihood on the
    training rows: a constant distribution, and a linear model with constant (co)variance."""
    out = {}
    A_tr = np.column_stack([np.ones(len(Xtr)), Xtr])
    A_te = np.column_stack([np.ones(len(Xte)), Xte])
    if ytr.ndim == 1:
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


def evaluate(model, Xte, yte):
    """Per-row log-density and, for a univariate response, the PIT values cdf(y | x)."""
    lp, pit = [], []
    for x, v in zip(Xte, yte):
        d = model(x)
        v = np.atleast_1d(v)[None] if np.ndim(v) else np.array([v])
        lp.append(float(np.asarray(d.logpdf(v))[0]))
        if np.ndim(yte) == 1:
            pit.append(float(np.asarray(d.cdf(v))[0]))
    return np.array(lp), (np.array(pit) if pit else None)


def plots(run_dir, model, X, y, Xte, yte, pit, cov_names, resp_names, seed=0):
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
        ax.set(xlabel="PIT = estimated cdf at the observed response", ylabel="count",
               title="Calibration on held-out rows")
        fig.savefig(run_dir / "calibration.png", dpi=150)
        plt.close(fig)
        files.append("calibration.png")
    if y.ndim == 1 and X.shape[1] == 1:
        gx = np.linspace(X[:, 0].min(), X[:, 0].max(), 150)
        pad = 0.1 * (y.max() - y.min())
        gy = np.linspace(y.min() - pad, y.max() + pad, 250)
        dens = np.array([np.exp(np.asarray(model(np.array([g])).logpdf(gy))) for g in gx])
        dens = dens / np.maximum(dens.max(axis=1, keepdims=True), 1e-300)
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ax.pcolormesh(gx, gy, dens.T, cmap="Blues", shading="auto", vmin=0, vmax=1)
        ax.scatter(X[:, 0], y, s=3, color="black", alpha=.35, linewidths=0, label="data")
        ax.set(xlabel=cov_names[0], ylabel=resp_names[0],
               title="Estimated conditional density (scaled to 1 at each x)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
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
         f"Training rows: {info['n_train']}; held-out rows: {info['n_test']}.", "",
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
              "Mean negative log-likelihood on the held-out rows (lower is better):", "",
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
