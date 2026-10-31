"""Reference program for counts.csv: the true model (for checking the pipeline)."""
from adr.dist import NegativeBinomial, exp


def model(x):
    return NegativeBinomial(exp(0.5 + 0.8 * x[0] - 0.5 * x[1]), 2.0)
