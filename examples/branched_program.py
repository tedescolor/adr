"""Two-branch Gaussian mixture: the program returned by ADR (Claude Opus 5.5) for one training
sample of this design in the paper (Program 2 of arXiv:2610.37611)."""
from adr.dist import Normal, Mixture, maximum


def model(x):
    t = x[0]
    up = ((-9.911676 * t + 19.820231) * t - 7.408399) * t + 0.001434
    lo = up - 8.394843 * maximum(t - 0.352661, 0.0) ** 2
    sd = maximum(0.400441 - 0.319172 * t, 0.02)
    return Mixture([0.5, 0.5], [Normal(up, sd), Normal(lo, sd)])
