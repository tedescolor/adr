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
fig.savefig(ROOT / "docs" / "example_data.png", dpi=150)
plt.close(fig)

# the estimate against the truth
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained",
                         gridspec_kw={"width_ratios": [1.15, 1]})
gx = np.linspace(df.x.min(), df.x.max(), 150)
gy = np.linspace(df.y.min() - .3, df.y.max() + .3, 250)
dens = np.array([np.exp(np.asarray(fit.distribution({"x": g}).logpdf(gy))) for g in gx])
axes[0].pcolormesh(gx, gy, (dens / dens.max(axis=1, keepdims=True)).T, cmap="Blues", shading="auto")
axes[0].scatter(df.x, df.y, s=3, color="black", alpha=.35, linewidths=0)
axes[0].set(xlabel="x", ylabel="y", title="Estimated conditional density (scaled at each x) and data")
yy = np.linspace(-2.2, 2.6, 600)
for k, x0 in enumerate([0.3, 0.8]):
    axes[1].plot(yy, true_density(yy, x0), color=f"C{k}", lw=2.2, alpha=.45, label=f"true, x = {x0}")
    axes[1].plot(yy, np.exp(fit.distribution({"x": x0}).logpdf(yy)), color=f"C{k}", lw=1.4, ls="--",
                 label=f"ADR, x = {x0}")
axes[1].set(xlabel="y", ylabel="conditional density", title="True and estimated densities")
axes[1].legend(frameon=False, fontsize=8)
fig.savefig(ROOT / "docs" / "example_fit.png", dpi=150)
plt.close(fig)
print("wrote docs/example_data.png and docs/example_fit.png")
