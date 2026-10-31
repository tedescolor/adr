"""Simulated example datasets (n = 1000 each), written next to this file.

- branched.csv (continuous): the univariate design of the ADR paper (Section 4). X ~ Unif(0, 1);
  for x < 0.5, Y | x ~ N(m1(x), s(x)^2); for x >= 0.5, an equal mixture of N(m1(x), s(x)^2) and
  N(m2(x), s(x)^2), with m1(x) = 10x(x-0.5)(1.5-x), m2(x) = 10x(x-0.5)(0.8-x), s(x) = 0.3(1.3-x).
- zero_inflated.csv (mixed): X ~ Unif(0, 1); Y = 0 with probability sigmoid(2 - 4x), otherwise
  Y ~ LogNormal(0.5 + x, 0.5).
- counts.csv (discrete): X1, X2 ~ N(0, 1); Y ~ negative binomial with mean
  exp(0.5 + 0.8 x1 - 0.5 x2) and size 2.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
n = 1000


def save(name, cols, header):
    np.savetxt(HERE / name, np.column_stack(cols), delimiter=",", header=header, comments="",
               fmt="%.6f")
    print(f"wrote {HERE / name}")


rng = np.random.default_rng(2026)
x = rng.uniform(0, 1, n)
m1 = 10 * x * (x - 0.5) * (1.5 - x)
m2 = 10 * x * (x - 0.5) * (0.8 - x)
m = np.where((x < 0.5) | (rng.random(n) < 0.5), m1, m2)
save("branched.csv", [x, m + 0.3 * (1.3 - x) * rng.standard_normal(n)], "x,y")

rng = np.random.default_rng(2027)
x = rng.uniform(0, 1, n)
zero = rng.random(n) < 1 / (1 + np.exp(-(2 - 4 * x)))
y = np.where(zero, 0.0, np.exp(0.5 + x + 0.5 * rng.standard_normal(n)))
save("zero_inflated.csv", [x, y], "x,y")

rng = np.random.default_rng(2028)
x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
mean = np.exp(0.5 + 0.8 * x1 - 0.5 * x2)
y = rng.negative_binomial(2, 2 / (2 + mean)).astype(float)
np.savetxt(HERE / "counts.csv", np.column_stack([x1, x2, y]), delimiter=",", header="x1,x2,y",
           comments="", fmt=["%.6f", "%.6f", "%d"])
print(f"wrote {HERE / 'counts.csv'}")
