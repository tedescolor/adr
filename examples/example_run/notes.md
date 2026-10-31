# Notes

Data: y follows one noisy trunk curve for x < ~0.6 (noise sd larger at small x), then splits
into two roughly equal-weight branches: an upper one rising to ~2.5 and a lower one falling to ~-1 at x=1.

Validation: 5-fold CV (fixed random split), MLE refitted per fold with BFGS + random restarts.

Candidates (mean CV NLL):
- A: 2-Normal mixture, independent polynomial means, log-sd polynomials, logistic weight in x:
  (3,1) 0.2196, (4,1) 0.2141, (4,2) 0.2204, (5,2) 0.2204
- B: shared trunk polynomial mu(x) (deg dm) + branch offsets a_k * g(x), where
  g = softplus split ramp at a learned point t0 (~0.63) with learned width; log-sd quadratic per
  component, constant weight:
  (3,1) 0.215, (4,1) 0.2185, (5,1) 0.2132, (5,2) 0.2107

- C: B(5,2) with Student-t components: nu -> ~1800 (Gaussian), CV 0.2097, no real gain; kept Normal.

Chosen: B(5,2): it builds in the branching structure (one curve before the split, two after) and
had the best CV. ~22 fitted constants. Train NLL 0.186, CV 0.2107.

expected_nll: 0.215
