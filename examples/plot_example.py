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

# the estimate against the truth: estimated density with the data, and true and estimated
# densities at two values of x
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained",
                         gridspec_kw={"width_ratios": [1.25, 1]})
gx = np.linspace(0, 1, 300)
gy = np.linspace(-2, 3, 400)
dens = np.array([np.exp(np.asarray(fit.distribution({"x": g}).logpdf(gy))) for g in gx])
ax = axes[0]
top = np.ceil(dens.max() * 4) / 4
cf = ax.contourf(gx, gy, dens.T, levels=np.linspace(0.02, top, 24), cmap="Blues")
for c in cf.collections if hasattr(cf, "collections") else []:
    c.set_edgecolor("face")
ax.scatter(df.x, df.y, s=4, color="black", alpha=.45, linewidths=0, label="data")
cols = {0.3: "#2a78d6", 0.8: "#d55181"}
for x0, c in cols.items():
    ax.axvline(x0, color=c, ls="--", lw=1.4)
ax.set(xlabel="x", ylabel="y", title="Estimated conditional density of y given x, with the data",
       xlim=(0, 1), ylim=(-2, 3))
cb = fig.colorbar(cf, ax=ax, shrink=.85, pad=.01)
cb.set_label("estimated density p(y | x)")
cb.set_ticks(np.arange(0.5, top + 1e-9, 0.5))
yy = np.linspace(-2, 3, 600)
for x0, c in cols.items():
    axes[1].plot(yy, true_density(yy, x0), color=c, lw=4, alpha=.3, label=f"true, x = {x0}")
    axes[1].plot(yy, np.exp(fit.distribution({"x": x0}).logpdf(yy)), color=c, lw=1.6,
                 label=f"ADR, x = {x0}")
axes[1].set(xlabel="y", ylabel="conditional density of y given x", ylim=(0, 2.0),
            title="True and estimated densities at x = 0.3 and x = 0.8")
axes[1].legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
fig.savefig(ROOT / "docs" / "example_fit.png", dpi=150)
plt.close(fig)
print("wrote docs/example_data.png and docs/example_fit.png")
