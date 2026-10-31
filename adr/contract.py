"""Check that a submitted model.py respects the output contract.

Usage: python -m adr.contract path/to/model.py [--features D] [--ydim Q] [--max-literals K]
                                                [--data data.csv] [--domain domain.csv]
                                                [--mode continuous|mixed|discrete]
                                                [--atoms A [A ...]] [--support SPEC]

With --mode mixed (point masses at the declared --atoms) or --mode discrete (values in the
declared --support), the checks use the score of adr.score.loglik instead of the log density:
draws must have a finite score, a mixed law may put mass only at the declared atoms, a discrete
law only on the support, and, with --data, every training row must have a finite score.

With --data, the runtime test uses 20 covariate rows of that csv (all columns except
``y``/``y*``) instead of random draws from U(-3, 3).

With --domain (a csv with columns ``column,min,max``, one row per covariate), the model must
also return a valid distribution at points of the covariate box: both corners, 50 uniform
draws in the box, and, starting from 5 of the test rows, each covariate set in turn to its
minimum and to its maximum.

The literal budget counts the constants of the formula: every numeric or boolean literal
and every use of ``pi`` or ``e`` (``math.pi`` too), but not integer indices such as the 3 in
``x[3]`` or ``x[0:5]``, which select a column rather than hold a value. String constants
(other than docstrings) are not allowed, so a constant cannot hide in text.
"""
import argparse
import ast
import importlib.util
import sys

import numpy as np

from adr import dist

ALLOWED_MODULES = {"adr.dist", "math"}
MAX_NUMERIC_LITERALS = 30
MAX_AST_NODES = 500
MAX_SEQUENCE_LEN = 6
FORBIDDEN_NODES = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp,
                   ast.GeneratorExp, ast.Global, ast.Nonlocal, ast.Try, ast.With,
                   ast.ClassDef, ast.Dict, ast.Set, ast.Yield, ast.Await)
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals",
                   "getattr", "setattr", "vars", "input", "print"}
CONSTANT_NAMES = {"pi", "e"}


def count_literals(tree):
    """Constants of the formula: numeric/boolean literals and uses of pi or e, excluding
    bare integer subscript indices, lists of them and slice bounds (x[3], x[-1], x[[0, 7]],
    x[0:5]), which select covariates; integers inside index arithmetic (T[k // 6]) count."""
    nodes = list(ast.walk(tree))

    def bare_int(c):  # 3 or -3
        if isinstance(c, ast.UnaryOp) and isinstance(c.op, ast.USub):
            c = c.operand
        return c if isinstance(c, ast.Constant) and type(c.value) is int else None

    index = set()
    for n in (n for n in nodes if isinstance(n, ast.Subscript)):
        parts = n.slice.elts if isinstance(n.slice, (ast.Tuple, ast.List)) else [n.slice]
        for p in parts:
            if isinstance(p, ast.Slice):
                bounds = [p.lower, p.upper, p.step]
            elif isinstance(p, (ast.List, ast.Tuple)):  # x[[0, 11, 15]]: a list of columns
                bounds = p.elts
            else:
                bounds = [p]
            index |= {id(c) for c in map(bare_int, filter(None, bounds)) if c is not None}
    n = 0
    for c in nodes:
        if isinstance(c, ast.Constant) and isinstance(c.value, (int, float, bool)):
            n += id(c) not in index
        elif isinstance(c, ast.Name) and c.id in CONSTANT_NAMES:
            n += 1
        elif isinstance(c, ast.Attribute) and c.attr in CONSTANT_NAMES:
            n += 1
    return n


def docstrings(tree):
    """ids of the docstring nodes of the module and its functions."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and \
                isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            out.add(id(n.body[0].value))
    return out


def check_source(src, max_literals=MAX_NUMERIC_LITERALS):
    problems = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        problems.append(f"program too large: {len(nodes)} AST nodes > {MAX_AST_NODES}")
    n_lit = count_literals(tree)
    if n_lit > max_literals:
        problems.append(f"too many numeric literals: {n_lit} > {max_literals}")

    docs = docstrings(tree)
    for n in nodes:
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and id(n) not in docs:
            problems.append(f"line {n.lineno}: string constants are not allowed (except docstrings)")
        elif isinstance(n, FORBIDDEN_NODES):
            problems.append(f"line {n.lineno}: forbidden construct {type(n).__name__}")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name not in ALLOWED_MODULES:
                    problems.append(f"line {n.lineno}: import of {a.name} not allowed")
        elif isinstance(n, ast.ImportFrom):
            if n.module not in ALLOWED_MODULES:
                problems.append(f"line {n.lineno}: import from {n.module} not allowed")
        elif isinstance(n, (ast.List, ast.Tuple)) and len(n.elts) > MAX_SEQUENCE_LEN:
            problems.append(f"line {n.lineno}: sequence literal longer than {MAX_SEQUENCE_LEN}")
        elif isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
            problems.append(f"line {n.lineno}: use of {n.id} not allowed")
        elif isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            problems.append(f"line {n.lineno}: private attribute access not allowed")

    if not any(isinstance(n, ast.FunctionDef) and n.name == "model" for n in tree.body):
        problems.append("no top-level function `model` defined")
    return problems


def load_model(path):
    spec = importlib.util.spec_from_file_location("submitted_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.model


def domain_points(lo, hi, base, rng):
    """Points of the covariate box [lo, hi] at which the model must be valid."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    pts = [lo, hi]
    pts += list(lo + (hi - lo) * rng.uniform(size=(50, len(lo))))
    for b in np.asarray(base, dtype=float)[:5]:
        for j in range(len(lo)):
            for v in (lo[j], hi[j]):
                x = b.copy()
                x[j] = v
                pts.append(x)
    return np.asarray(pts)


def valid_at(model, x, rng, y_dim, mode="continuous", atoms=(), support=None):
    """None if model(x) is a valid distribution with finite log-density at its own samples."""
    try:
        d = model(x)
    except Exception as e:
        return f"model(x) raised {type(e).__name__}: {e}"
    if not isinstance(d, dist.Distribution):
        return f"model(x) returned {type(d).__name__}, not adr.dist.Distribution"
    try:
        ys = np.asarray(d.sample(rng, 5))
        expected = (5,) if y_dim == 1 else (5, y_dim)
        if ys.shape != expected:
            return f"sample(rng, 5) has shape {ys.shape}, expected {expected}"
        if mode == "continuous":
            lp = np.asarray(d.logpdf(ys))
        else:
            from adr.score import loglik
            lp = loglik(d, ys, mode, atoms, support)
            if mode == "mixed":
                from adr.score import TOL
                a = np.asarray(atoms, dtype=float)
                more = np.asarray(d.sample(rng, 200), dtype=float)
                vals, counts = np.unique(more, return_counts=True)
                extra = [v for v, c in zip(vals, counts) if c > 1 and
                         not np.any(np.abs(v - a) <= TOL * np.maximum(1, np.abs(a)))]
                if extra:
                    return f"point mass at {extra[0]:g}, which is not a declared atom"
            if mode == "discrete" and not np.all(support.contains(ys)):
                return "draws outside the declared support"
    except Exception as e:
        return f"sampling or logpdf raised {type(e).__name__}: {e}"
    if lp.shape != (5,) or not np.all(np.isfinite(lp)):
        return "logpdf of the model's own samples is not finite"
    return None


def check(path, n_features=1, y_dim=1, max_literals=MAX_NUMERIC_LITERALS, xs=None, domain=None,
          mode="continuous", atoms=(), support=None, ys=None):
    with open(path) as f:
        problems = check_source(f.read(), max_literals)
    if problems:
        return problems
    model = load_model(path)
    rng = np.random.default_rng(0)
    all_xs_full = None if xs is None else np.asarray(xs, dtype=float)
    if xs is None:
        xs = rng.uniform(-3, 3, size=(20, n_features))
    else:
        xs = np.asarray(xs, dtype=float)
        xs = xs[rng.choice(len(xs), size=min(20, len(xs)), replace=False)]
    for x in xs:
        err = valid_at(model, x, rng, y_dim, mode, atoms, support)
        if err:
            return [err]
    if mode != "continuous" and ys is not None and all_xs_full is not None:
        from adr.score import loglik
        for i, (x, v) in enumerate(zip(all_xs_full, ys)):
            try:
                ok = np.all(np.isfinite(loglik(model(x), [v], mode, atoms, support)))
            except Exception as e:
                return [f"training row {i}: {type(e).__name__}: {e}"]
            if not ok:
                return [f"training row {i} (y = {v:g}) has probability zero under the model"]
    if domain is not None:
        lo, hi = domain
        for x in domain_points(lo, hi, xs, rng):
            err = valid_at(model, x, rng, y_dim, mode, atoms, support)
            if err:
                shown = ", ".join(f"x{j}={v:.6g}" for j, v in enumerate(x))
                return [f"in the covariate domain, at ({shown}): {err}"]
    return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--features", type=int, default=1)
    ap.add_argument("--ydim", type=int, default=1)
    ap.add_argument("--max-literals", type=int, default=MAX_NUMERIC_LITERALS)
    ap.add_argument("--data", help="csv whose non-y columns supply the test covariates")
    ap.add_argument("--domain", help="csv (column,min,max) giving the box of covariate values")
    ap.add_argument("--mode", choices=["continuous", "mixed", "discrete"], default="continuous")
    ap.add_argument("--atoms", type=float, nargs="+", default=())
    ap.add_argument("--support", default=None)
    a = ap.parse_args()
    xs = ys = None
    support = None
    if a.support:
        from adr.score import Support
        support = Support.parse(a.support)
    if a.data:
        import pandas as pd
        df = pd.read_csv(a.data)
        xs = df[[c for c in df.columns if not c.startswith("y")]].to_numpy()
        if "y" in df.columns:
            ys = df["y"].to_numpy(dtype=float)
    domain = None
    if a.domain:
        import pandas as pd
        dm = pd.read_csv(a.domain)
        domain = (dm["min"].to_numpy(float), dm["max"].to_numpy(float))
    probs = check(a.path, a.features, a.ydim, a.max_literals, xs, domain, a.mode, a.atoms,
                  support, ys)
    print("OK: contract satisfied" if not probs else "REJECTED:\n  " + "\n  ".join(probs))
    sys.exit(1 if probs else 0)
