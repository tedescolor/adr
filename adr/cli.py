"""Command line of ADR.

    adr fit DATA.csv --response Y [Y2] [--covariates A B ...] [--model claude-opus-5-5]
            [--budget 60] [--test-fraction 0.2 | --test TEST.csv] [--out DIR] [--seed 0]
            [--keep-names] [--sandbox] [--no-domain] [--dry-run]
            [--mixed [--atoms A ...] | --discrete [--support SPEC]]
    adr inspect DATA.csv --response Y [--covariates ...]
    adr report RUN_DIR
    adr check MODEL.py --data DATA.csv --response Y [Y2] [--covariates ...]
"""
import argparse
import datetime
import json
import os
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # no __pycache__ next to the returned programs
DEFAULT_MODEL = "claude-opus-5-5"


def read_data(path, response, covariates=None):
    """Numeric covariate matrix and response from a csv; rows with missing values dropped."""
    import pandas as pd
    df = pd.read_csv(path)
    missing = [c for c in response + (covariates or []) if c not in df.columns]
    if missing:
        sys.exit(f"columns not found in {path}: {missing}; available: {list(df.columns)}")
    covariates = covariates or [c for c in df.columns if c not in response]
    sub = df[covariates + response].copy()
    for c in sub.columns:
        if sub[c].dtype == bool:
            sub[c] = sub[c].astype(float)
    bad = [c for c in sub.columns if not np.issubdtype(sub[c].dtype, np.number)]
    if bad:
        sys.exit(f"non-numeric columns {bad}: encode them as numbers (for example 0/1 "
                 f"indicators) or leave them out with --covariates")
    n0 = len(sub)
    sub = sub.dropna()
    if len(sub) < n0:
        print(f"dropped {n0 - len(sub)} rows with missing values")
    X = sub[covariates].to_numpy(float)
    y = sub[response].to_numpy(float)
    return X, (y[:, 0] if len(response) == 1 else y), covariates


def response_type(a, ytr, yte):
    """Response type of a fit (continuous, mixed or discrete), with its atoms or support."""
    from adr.score import Support, analyse_response, detect_atoms
    mixed, discrete = a.mixed or bool(a.atoms), a.discrete or bool(a.support)
    if mixed and discrete:
        sys.exit("choose one response type: --mixed/--atoms or --discrete/--support")
    if (mixed or discrete) and ytr.ndim != 1:
        sys.exit("mixed and discrete responses must be univariate")
    if mixed:
        atoms = [float(v) for v in a.atoms] if a.atoms else detect_atoms(ytr)
        if not atoms:
            sys.exit("no point mass found (no value in at least 1% of the training rows, and at "
                     "least 5 of them); give the atoms with --atoms, or fit a continuous response")
        print("response type: mixed, point masses at " + ", ".join(f"{v:g}" for v in atoms)
              + ("" if a.atoms else " (detected on the training rows)"), flush=True)
        return "mixed", atoms, None
    if discrete:
        if a.support:
            support = Support.parse(a.support)
        else:
            spec = analyse_response(ytr).get("support")
            if spec is None:
                sys.exit("could not find a grid containing the responses; give it with --support")
            support = Support.parse(spec)
        bad = ytr[~support.contains(ytr)]
        if len(bad):
            sys.exit(f"{len(bad)} training responses are outside the support {support.spec()} "
                     f"(for example {bad[0]:g})")
        if yte is not None and not support.contains(yte).all():
            print("warning: some held-out responses are outside the support; their score is infinite")
        print(f"response type: discrete, on {support.describe()} (--support {support.spec()})"
              + ("" if a.support else "; if the response has a maximum, give it with --support"),
              flush=True)
        return "discrete", [], support
    return "continuous", [], None


def inspect(a):
    """General description of the data and suggested response type."""
    import pandas as pd
    from adr.score import analyse_response
    df = pd.read_csv(a.data)
    if any(c not in df.columns for c in a.response):
        sys.exit(f"response not found; available columns: {list(df.columns)}")
    cov = a.covariates or [c for c in df.columns if c not in a.response]
    print(f"{a.data}: {len(df)} rows, {len(df.columns)} columns\n")
    print("Covariates:")
    for c in cov:
        s = df[c]
        if np.issubdtype(s.dtype, np.number) or s.dtype == bool:
            v = s.astype(float)
            kind = "constant" if v.nunique() <= 1 else ("binary" if v.nunique() == 2 else "numeric")
            print(f"  {c}: {kind}, {v.nunique()} distinct values, range [{v.min():g}, {v.max():g}]"
                  + (f", {s.isna().sum()} missing" if s.isna().any() else ""))
        else:
            print(f"  {c}: NOT NUMERIC ({s.nunique()} distinct values): encode it as numbers "
                  "(for example 0/1 indicators) or leave it out with --covariates")
    out = {"rows": len(df), "covariates": cov, "response": a.response}
    if len(a.response) == 2:
        print(f"\nResponse: bivariate ({', '.join(a.response)}); only the continuous type is "
              "available for bivariate responses.")
        out["type"] = "continuous"
        cmd = f"adr fit {a.data} -y {' '.join(a.response)}"
    else:
        y = df[a.response[0]]
        if not np.issubdtype(y.dtype, np.number):
            sys.exit(f"the response {a.response[0]} is not numeric")
        y = y.dropna().to_numpy(float)
        r = analyse_response(y)
        out.update(r)
        print(f"\nResponse {a.response[0]}: {r['distinct']} distinct values in {r['n']} rows, "
              f"range [{r['min']:g}, {r['max']:g}], {r['share_tied']:.1%} of the rows share "
              "their value with another row")
        if r["most_frequent"]:
            print("  most frequent values: " + ", ".join(
                f"{v:g} ({c}, {c / r['n']:.1%})" for v, c in r["most_frequent"]))
        if r["grid_step"] is not None:
            print(f"  all values lie on a grid with step {r['grid_step']:g}")
        cmd = f"adr fit {a.data} -y {a.response[0]}"
        if r["type"] == "mixed":
            cmd += " --mixed"
        elif r["type"] == "discrete":
            cmd += f" --discrete --support {r['support']}"
        print(f"\nSuggested response type: {r['type'].upper()}"
              + (" with point masses at " + ", ".join(f"{v:g}" for v in r["atoms"])
                 if r["type"] == "mixed" else "")
              + (f" on --support {r['support']}" if r["type"] == "discrete" else ""))
        for n in r["notes"]:
            print("  - " + n)
    if a.covariates:
        cmd += " -x " + " ".join(a.covariates)
    out["command"] = cmd
    print(f"\nSuggested command:\n  {cmd}")
    if a.json:
        print("\nJSON: " + json.dumps(out))


def fit(a):
    from adr import agent, evaluate
    from adr.contract import check, load_model
    if not 1 <= len(a.response) <= 2:
        sys.exit("--response takes one column (univariate) or two columns (bivariate)")
    X, y, cov = read_data(a.data, a.response, a.covariates)
    if a.test:
        Xte, yte, _ = read_data(a.test, a.response, cov)
        Xtr, ytr = X, y
    elif a.test_fraction > 0:
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(y))
        k = int(round(a.test_fraction * len(y)))
        Xte, yte, Xtr, ytr = X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]
    else:
        Xtr, ytr, Xte, yte = X, y, None, None
    if len(ytr) < 20:
        sys.exit("too few training rows (at least 20 are needed)")
    mode, atoms, support = response_type(a, ytr, yte)
    stem = Path(a.data).stem
    out = Path(a.out or f"adr_runs/{stem}-{datetime.datetime.now():%Y%m%d-%H%M%S}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    full_X = X if Xte is None or a.test is None else np.vstack([X, Xte])
    domain = None if a.no_domain else np.column_stack([full_X.min(0), full_X.max(0)])
    # The agent sees the path of its working directory, so the task is prepared in a neutral
    # location (a random name, no file or column names) and moved to <out>/task afterwards.
    neutral = not (a.dry_run or a.program)
    task_dir = (Path(os.environ.get("ADR_WS", "/var/tmp/adr")) / secrets.token_hex(6) / "task"
                if neutral else out / "task")
    task = agent.prepare_task(task_dir, Xtr, ytr, a.max_literals, sys.executable, a.budget,
                              domain, cov + a.response if a.keep_names else None,
                              mode, atoms, support)
    info = {"data": str(a.data), "response": a.response, "covariates": cov,
            "n_train": int(len(ytr)), "n_test": 0 if yte is None else int(len(yte)),
            "budget_min": a.budget, "seed": a.seed, "keep_names": a.keep_names,
            "domain": None if domain is None else domain.tolist(), "mode": mode,
            "atoms": atoms, "support": None if support is None else support.spec(),
            "support_text": None if support is None else support.describe()}
    if a.dry_run:
        (out / "run.json").write_text(json.dumps(info, indent=1))
        print(f"task prepared in {task} (dry run: no agent started)")
        return
    if a.program:  # evaluate a given program instead of running an agent
        shutil.copy(a.program, task / "model.py")
        info["agent"] = {"status": "not run (program given with --program)", "model": None}
        info["audit"] = []
    else:
        settings = agent.sandbox_settings(task) if a.sandbox else None
        print(f"starting the agent ({a.model}, budget {a.budget} min) in {task} ...", flush=True)
        info["agent"] = agent.run_agent(task, a.model, a.budget, settings)
        print(f"agent {info['agent']['status']} after {info['agent']['minutes']} min", flush=True)
        info["audit"] = agent.audit(task)
        shutil.move(str(task), str(out / "task"))
        shutil.rmtree(task.parent, ignore_errors=True)
        task = out / "task"
    prog = task / "model.py"
    if prog.exists():
        info["contract"] = check(prog, Xtr.shape[1], 1 if ytr.ndim == 1 else 2, a.max_literals,
                                 xs=Xtr, domain=None if domain is None else (domain[:, 0], domain[:, 1]),
                                 mode=mode, atoms=atoms, support=support, ys=ytr) or "ok"
    else:
        info["contract"] = ["no model.py written"]
    for f in ("model.py", "notes.md"):
        if (task / f).exists():
            shutil.copy(task / f, out / f)
    info["evaluation"], info["figures"] = None, []
    if info["contract"] == "ok":
        model = load_model(out / "model.py")
        if yte is not None:
            lp, pit = evaluate.evaluate(model, Xte, yte, mode, atoms, support)
            ev = {"nll": float(-lp.mean()),
                  "reference": evaluate.reference_nll(Xtr, ytr, Xte, yte, mode, atoms, support)}
            if pit is not None:
                ev["coverage"] = {"50": float(np.mean((pit > .25) & (pit < .75))),
                                  "90": float(np.mean((pit > .05) & (pit < .95)))}
            info["evaluation"] = ev
        else:
            pit = None
        try:
            info["figures"] = evaluate.plots(out, model, X, y, Xte if Xte is not None else Xtr,
                                             yte if yte is not None else ytr, pit, cov, a.response,
                                             mode=mode, atoms=atoms, support=support)
        except Exception as e:  # plots are optional
            print(f"plots skipped: {e}")
    (out / "run.json").write_text(json.dumps(info, indent=1))
    rep = evaluate.report(out)
    print(f"report: {rep}")
    if info["evaluation"]:
        print(f"held-out NLL: ADR {info['evaluation']['nll']:.4f}; " + "; ".join(
            f"{k} {v:.4f}" for k, v in info["evaluation"]["reference"].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="adr", description="Agentic distributional regression")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="estimate a conditional density with an LLM agent")
    f.add_argument("data", help="csv file with a header row")
    f.add_argument("--response", "-y", nargs="+", required=True,
                   help="response column (two columns for a bivariate response)")
    f.add_argument("--covariates", "-x", nargs="+", default=None,
                   help="covariate columns (default: all other columns)")
    f.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default {DEFAULT_MODEL})")
    f.add_argument("--budget", type=int, default=60, help="time budget in minutes (default 60)")
    f.add_argument("--test-fraction", type=float, default=0.2,
                   help="fraction of rows held out for evaluation (default 0.2; 0 = none)")
    f.add_argument("--test", help="separate csv of held-out rows (instead of --test-fraction)")
    f.add_argument("--out", help="output directory (default adr_runs/<data>-<time>)")
    f.add_argument("--seed", type=int, default=0, help="seed of the train/held-out split")
    f.add_argument("--max-literals", type=int, default=80, help="contract: maximum constants")
    f.add_argument("--keep-names", action="store_true",
                   help="tell the agent the column names (default: anonymous columns)")
    f.add_argument("--sandbox", action="store_true",
                   help="run the agent's shell in the Claude Code sandbox (no network)")
    f.add_argument("--no-domain", action="store_true",
                   help="do not require validity on the box of observed covariate values")
    f.add_argument("--mixed", action="store_true",
                   help="the response has point masses (atoms detected on the training rows "
                        "unless given with --atoms)")
    f.add_argument("--atoms", type=float, nargs="+", default=None,
                   help="values with positive probability (implies --mixed)")
    f.add_argument("--discrete", action="store_true",
                   help="the response is discrete (support detected unless given with --support)")
    f.add_argument("--support", default=None,
                   help="support of a discrete response: naturals, integers, a:b, a:b:step, "
                        "a:inf, or v1,v2,... (implies --discrete)")
    f.add_argument("--dry-run", action="store_true", help="prepare the task without the agent")
    f.add_argument("--program", help="evaluate this model.py instead of running an agent")
    i = sub.add_parser("inspect", help="describe the data and suggest the response type")
    i.add_argument("data")
    i.add_argument("--response", "-y", nargs="+", required=True)
    i.add_argument("--covariates", "-x", nargs="+", default=None)
    i.add_argument("--json", action="store_true", help="also print the result as JSON")
    r = sub.add_parser("report", help="rewrite report.md of a run directory")
    r.add_argument("run_dir")
    c = sub.add_parser("check", help="check a program against the output contract")
    c.add_argument("model")
    c.add_argument("--data", required=True)
    c.add_argument("--response", "-y", nargs="+", required=True)
    c.add_argument("--covariates", "-x", nargs="+", default=None)
    c.add_argument("--max-literals", type=int, default=80)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(a)
    elif a.cmd == "inspect":
        inspect(a)
    elif a.cmd == "report":
        from adr.evaluate import report
        print(report(a.run_dir))
    elif a.cmd == "check":
        from adr.contract import check
        X, y, _ = read_data(a.data, a.response, a.covariates)
        probs = check(a.model, X.shape[1], 1 if y.ndim == 1 else 2, a.max_literals, xs=X,
                      domain=(X.min(0), X.max(0)))
        print("OK: contract satisfied" if not probs else "REJECTED:\n  " + "\n  ".join(probs))
        sys.exit(1 if probs else 0)


if __name__ == "__main__":
    main()
