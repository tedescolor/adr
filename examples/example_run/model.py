from adr.dist import Mixture, Normal, exp, log, sigmoid, maximum, abs


def model(x):
    t = x[0]
    z = (t - 0.634068) / 0.087396
    g = 0.087396 * (maximum(z, 0.0) + log(1.0 + exp(-abs(z))))
    mu = (((((-47.466309) * t + 108.978164) * t + -86.112679) * t + 42.877879) * t + -9.908462) * t + -0.003241
    sd1 = exp(((-1.325764) * t + -0.179717) * t + -0.994224)
    sd2 = exp(((-0.858171) * t + -0.484573) * t + -0.935113)
    w = sigmoid(0.114519)
    return Mixture([w, 1.0 - w], [Normal(mu + -16.056164 * g, sd1), Normal(mu + -25.489746 * g, sd2)])

