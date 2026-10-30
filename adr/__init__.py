"""Agentic distributional regression (ADR): conditional density estimation through
agent-written programs (Tedesco and Finazzi, arXiv:2610.37611)."""
__version__ = "0.1.0"


def load(run_dir):
    """Load the result of `adr fit` (its output directory); see adr.fitted.FittedADR."""
    from adr.fitted import load as _load
    return _load(run_dir)
