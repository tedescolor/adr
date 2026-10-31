"""Figures of the README example (docs/example_data.png, docs/example_fit.png): the data of
examples/branched.csv, and the estimate of the run in examples/example_run against the true
conditional density. Run from the repository root."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import adr

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "examples" / "branched.csv")
fit = adr.load(ROOT / "examples" / "example_run")


def true_density(y, x):
    m1, m2 = 10 * x * (x - .5) * (1.5 - x), 10 * x * (x - .5) * (.8 - x)
    s = .3 * (1.3 - x)
    return stats.norm.pdf(y, m1, s) if x < .5 else .5 * (stats.norm.pdf(y, m1, s) + stats.norm.pdf(y, m2, s))


# the data
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.scatter(df.x, df.y, s=5, color="0.25", alpha=.5, linewidths=0)
ax.set(xlabel="x", ylabel="y", title="examples/branched.csv: 1,000 observations")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(ROOT / "docs" / "example_data.png", dpi=150)
plt.close(fig)

# the estimate against the truth: data, draws from the estimate at the same x, and densities
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained",
                         gridspec_kw={"width_ratios": [1, 1, 1.1]})
rng = np.random.default_rng(1)
sim = np.array([float(fit.distribution({"x": v}).sample(rng, 1)[0]) for v in df.x])
cols = {0.3: "#2a78d6", 0.8: "#d55181"}
for ax, yv, title in zip(axes[:2], (df.y, sim), ("Observed data", "Simulated from the ADR estimate")):
    ax.scatter(df.x, yv, s=5, color="0.3", alpha=.45, linewidths=0)
    for x0, c in cols.items():
        ax.axvline(x0, color=c, ls="--", lw=1.2)
    ax.set(xlabel="x", ylabel="y", title=title, ylim=(-2, 3))
axes[1].sharey(axes[0])
yy = np.linspace(-2, 3, 600)
for x0, c in cols.items():
    axes[2].plot(yy, true_density(yy, x0), color=c, lw=4, alpha=.3, label=f"true, x = {x0}")
    axes[2].plot(yy, np.exp(fit.distribution({"x": x0}).logpdf(yy)), color=c, lw=1.6,
                 label=f"ADR, x = {x0}")
axes[2].set(xlabel="y", ylabel="conditional density of y given x",
            title="True and estimated densities")
axes[2].set_ylim(0, 2.0)
axes[2].legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
fig.savefig(ROOT / "docs" / "example_fit.png", dpi=150)
plt.close(fig)
print("wrote docs/example_data.png and docs/example_fit.png")
