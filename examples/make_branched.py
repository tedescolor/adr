"""Simulated example: the univariate design of the ADR paper (Section 4).

X ~ Unif(0, 1); for x < 0.5, Y | x ~ N(m1(x), s(x)^2); for x >= 0.5, Y | x is an equal mixture
of N(m1(x), s(x)^2) and N(m2(x), s(x)^2), with m1(x) = 10x(x-0.5)(1.5-x),
m2(x) = 10x(x-0.5)(0.8-x) and s(x) = 0.3(1.3-x). Writes examples/branched.csv (n = 1000).
"""
from pathlib import Path

import numpy as np

rng = np.random.default_rng(2026)
n = 1000
x = rng.uniform(0, 1, n)
m1 = 10 * x * (x - 0.5) * (1.5 - x)
m2 = 10 * x * (x - 0.5) * (0.8 - x)
m = np.where((x < 0.5) | (rng.random(n) < 0.5), m1, m2)
y = m + 0.3 * (1.3 - x) * rng.standard_normal(n)
out = Path(__file__).with_name("branched.csv")
np.savetxt(out, np.column_stack([x, y]), delimiter=",", header="x,y", comments="", fmt="%.6f")
print(f"wrote {out}")
