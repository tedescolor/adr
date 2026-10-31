"""Reference program for zero_inflated.csv: the true model (for checking the pipeline)."""
from adr.dist import Mixture, PointMass, LogNormal, sigmoid


def model(x):
    p0 = sigmoid(2.0 - 4.0 * x[0])
    return Mixture([p0, 1.0 - p0], [PointMass(0.0), LogNormal(0.5 + x[0], 0.5)])
