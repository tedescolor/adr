"""Prepare an ADR task and run one isolated LLM agent on it.

The task directory contains only the training data (columns renamed ``x0``.., ``y`` or
``y0``, ``y1``), a copy of the distribution library (``adr/dist.py``, ``adr/contract.py``),
``TASK.md`` and, optionally, ``domain.csv``. The agent is a headless Claude Code session
(``claude -p``) started in that directory with a fixed prompt, restricted tools and a
wall-clock budget; it never sees the original column names, the file name or any
description of the data unless ``column_names`` is given. This is the protocol of the
paper (Tedesco and Finazzi, arXiv:2610.37611).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

PROMPT = ("You are a statistician solving a conditional density estimation task. Your working "
          "directory is {dir}. Read TASK.md there and follow it exactly: it describes the data, "
          "the output contract, the rules (work only inside that directory, OMP_NUM_THREADS=1, at "
          "most 2 processes) and the deliverables (model.py passing the contract check, and "
          "notes.md ending with an `expected_nll: <value>` line). Take the time you need to "
          "explore the data, compare candidate programs with honest out-of-sample validation, "
          "and refine. When done, reply with a short summary: the final program's structure, "
          "number of constants, your validation NLL and expected_nll.")

_COMMON = """
**Contract** (read `adr/dist.py` and `adr/contract.py`):
- `model.py` imports only from `adr.dist` and `math`; `model` is a pure function with its
  constants written in as numbers;
- at most {max_literals} constants and 500 AST nodes. Constants are counted as: every
  numeric or boolean literal and every use of `pi` or `e`; integer column indices such as
  the 3 in `x[3]` are not counted. String constants are not allowed (except docstrings);
- no loops, comprehensions, dicts or sets; no list/tuple literal longer than 6;
- check it with
  `{check}`
  (a model that fails the check scores nothing).

**Rules** (every action is logged and audited).
- Use only `data.csv` and your own analysis. No internet: do not use web search or fetch
  tools, do not download or install anything, do not call any network service.
- Work only inside this directory: put scratch files in `./work/`, not in /tmp or any
  scratch/temp directory, and do not read, list or search anything outside this directory.
- Do not try to identify the dataset or find other copies of it; treat it as anonymous.
- Use `{python}` (numpy, scipy, pandas, matplotlib). Run with `OMP_NUM_THREADS=1` and
  at most 2 processes at a time.

**Time budget: {budget} minutes** of wall-clock time from the start. Save a valid `model.py`
(and `notes.md`) early and overwrite them as you improve; at {budget} minutes the run is stopped
and whatever `model.py` is on disk is scored.

**Deliverables** (in this directory):
- `model.py`, passing the check;
- `notes.md`: what you tried, the validation scheme, why you chose the final program, and
  your estimate of its mean NLL on new data (one number, on its own line as
  `expected_nll: <value>`).
"""

TASK_UNIVARIATE = """# Task: conditional density estimation

`data.csv` has {n} rows: {covs} and a continuous target `y`.
Write `model.py` defining `model(x)` that, for one covariate row `x` (a numpy array of
length {d}, in the column order of `data.csv`), returns an `adr.dist.Distribution` for `y`.

**How you are judged.** The mean negative log-likelihood (natural log) of `model` on
held-out rows from the same distribution, which you do not have. Lower is better.
The density must be on the original scale of `y`: if you transform `y` while
exploring, the returned distribution must still be a density for the untransformed `y`.
Overfitting `data.csv` is your responsibility; how you validate is up to you.
""" + _COMMON

TASK_BIVARIATE = """# Task: conditional density estimation (bivariate response)

`data.csv` has {n} rows: {covs} and a bivariate response `y0`, `y1`.
Write `model.py` defining `model(x)` that, for one covariate row `x` (a numpy array of
length {d}, in the column order of `data.csv`), returns an `adr.dist.Distribution` for the
pair `(y0, y1)`: its `logpdf` takes an array of shape (n, 2) and its `sample(rng, n)` returns
an array of shape (n, 2). `adr/dist.py` explains how to build bivariate distributions
(`MultivariateNormal`, `Independent`, `Sequential`, `Mixture`).

**How you are judged.** The mean negative log-likelihood (natural log) of `model`, as a
joint density for `(y0, y1)`, on held-out rows from the same distribution, which you do not
have. Lower is better. Overfitting `data.csv` is your responsibility; how you validate is up
to you.
""" + _COMMON

JUDGED_MIXED = """**Response type: continuous with point masses.** `y` can be exactly equal to {atoms}:
P(Y = a | x) can be positive at these atoms, while all the other values of `y` come from a
continuous distribution.

**How you are judged.** The mean score on held-out rows from the same distribution, which you
do not have; lower is better. For a row whose `y` equals one of the atoms, the score is
-log P(Y = y | x); for any other `y`, it is -log of the density of the continuous part of the
distribution. `adr.task.loglik(d, y)` computes the log-likelihood used for this score. Build the
atoms with `PointMass(a)` inside `Mixture` (the weights may depend on x); point masses at other
values are not allowed. Overfitting `data.csv` is your responsibility; how you validate is up to
you.
"""

JUDGED_DISCRETE = """**Response type: discrete.** `y` takes values in {support}.

**How you are judged.** The mean of -log P(Y = y | x) on held-out rows from the same
distribution, which you do not have; lower is better. `adr.task.loglik(d, y)` computes this
log-likelihood. The distribution must put all its probability on the values above: use the
discrete families of `adr/dist.py` (`Poisson`, `NegativeBinomial`, `Binomial`, `BetaBinomial`,
`Categorical`, `PointMass`, `Shift`, and `Mixture` of these). Overfitting `data.csv` is your
responsibility; how you validate is up to you.
"""

TASK_SCORE = '''"""The score of this task: ``loglik(d, y)`` is the log-likelihood used to judge model.py."""
from adr.score import Support, loglik as _loglik

MODE = {mode!r}
ATOMS = {atoms!r}
SUPPORT = {support}


def loglik(d, y):
    return _loglik(d, y, MODE, ATOMS, SUPPORT)
'''

DOMAIN_TXT = """
**Covariate domain.** `domain.csv` gives, for each covariate, the smallest and the largest value
in the full dataset, including the held-out rows on which `model` is scored. Every row on which
`model` is evaluated lies in this box. `model(x)` must return a valid distribution for every `x`
in the box, not only for the rows of `data.csv`: the contract check also evaluates it at points
of the box, including each covariate at its minimum and at its maximum.
"""

TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
DISALLOWED = ["WebFetch", "WebSearch", "Agent", "Task"]
RESUME = ("The previous turn ended with an error ({error}). Continue the task from where you "
          "stopped; the rules and deliverables in TASK.md are unchanged.")
MAX_ATTEMPTS = 20


def prepare_task(workdir, X, y, max_literals=80, python=sys.executable, budget_min=60,
                 domain=None, column_names=None, mode="continuous", atoms=(), support=None):
    """Write data.csv, the library, TASK.md and (optionally) domain.csv into workdir.

    X: (n, d) covariates; y: (n,) or (n, 2) response. domain: (d, 2) array of [min, max] per
    covariate, enforced by the contract check. column_names: optional original names
    (covariates then response), disclosed to the agent only if given. mode: response type
    (continuous, mixed with ``atoms``, or discrete on ``support``, an adr.score.Support); a
    continuous task receives the library and contract of the paper unchanged."""
    import pandas as pd
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
    bivariate = y.ndim == 2
    if bivariate:
        df["y0"], df["y1"] = y[:, 0], y[:, 1]
    else:
        df["y"] = y
    df.to_csv(workdir / "data.csv", index=False)
    lib = workdir / "adr"
    lib.mkdir(exist_ok=True)
    if mode == "continuous":
        for f in ("dist.py", "contract.py"):
            shutil.copy(HERE / "_paper" / f, lib / f)
    else:
        if bivariate:
            raise ValueError("mixed and discrete responses must be univariate")
        for f in ("dist.py", "contract.py", "score.py"):
            shutil.copy(HERE / f, lib / f)
        (lib / "task.py").write_text(TASK_SCORE.format(
            mode=mode, atoms=[float(a) for a in atoms],
            support="None" if support is None else f"Support.parse({support.spec()!r})"))
    (lib / "__init__.py").write_text("")
    check = (f"{python} -m adr.contract model.py --features {X.shape[1]}"
             + (" --ydim 2" if bivariate else "") + f" --max-literals {max_literals} --data data.csv")
    if mode == "mixed":
        check += " --mode mixed --atoms " + " ".join(f"{a:g}" for a in atoms)
    elif mode == "discrete":
        check += f" --mode discrete --support {support.spec()}"
    if domain is not None:
        domain = np.asarray(domain, dtype=float)
        pd.DataFrame({"column": [f"x{j}" for j in range(len(domain))], "min": domain[:, 0],
                      "max": domain[:, 1]}).to_csv(workdir / "domain.csv", index=False)
        check += " --domain domain.csv"
    task = (TASK_BIVARIATE if bivariate else TASK_UNIVARIATE).format(
        n=len(y), d=X.shape[1], covs=("1 covariate `x0`" if X.shape[1] == 1 else
            f"{X.shape[1]} covariates `x0`..`x{X.shape[1] - 1}`"), max_literals=max_literals,
        python=python, budget=budget_min, check=check)
    if mode != "continuous":
        start = task.index("**How you are judged.**")
        end = task.index("\n**Contract**")
        judged = (JUDGED_MIXED.format(atoms=", ".join(f"{a:g}" for a in atoms)) if mode == "mixed"
                  else JUDGED_DISCRETE.format(support=support.describe()))
        task = task[:start] + judged + task[end:]
        task = task.replace("and a continuous target `y`", "and a target `y`")
    if domain is not None:
        task = task.replace("\n**Rules**", DOMAIN_TXT + "\n**Rules**", 1)
    if column_names is not None:
        d = X.shape[1]
        ys = ["y0", "y1"] if bivariate else ["y"]
        task += "\n**Column meanings.** " + ", ".join(
            [f"`x{j}` = {c}" for j, c in enumerate(column_names[:d])]
            + [f"`{t}` = {c}" for t, c in zip(ys, column_names[d:])]) + ".\n"
    (workdir / "TASK.md").write_text(task)
    return workdir


def sandbox_settings(workdir):
    """Claude Code settings enabling the Bash sandbox (requires bubblewrap on Linux or
    the macOS sandbox): no network, writes confined to the task directory, reads confined
    to the task directory and the Python installation."""
    workdir = Path(workdir).resolve()
    home = str(Path.home())
    allow_read = sorted({str(workdir), sys.prefix, sys.base_prefix, sys.exec_prefix})
    settings = {"sandbox": {
        "enabled": True, "autoAllowBashIfSandboxed": True, "failIfUnavailable": True,
        "allowUnsandboxedCommands": False,
        "network": {"allowedDomains": [], "allowUnixSockets": [], "allowAllUnixSockets": False,
                    "allowLocalBinding": False},
        "filesystem": {"allowWrite": [str(workdir)], "denyRead": [home, "/tmp"],
                       "allowRead": allow_read}}}
    path = workdir.parent / "sandbox_settings.json"
    path.write_text(json.dumps(settings, indent=1))
    return path


def agent_flags(model=None, settings=None):
    """Flags of the agent session: only the tools in TOOLS, file tools confined to the working
    directory, no user/project settings, MCP servers, skills or web tools."""
    flags = ["--tools", *TOOLS, "--allowedTools", *TOOLS, "--disallowedTools", *DISALLOWED,
             "--restricted", "--strict-mcp-config", "--disable-slash-commands",
             "--output-format", "stream-json", "--verbose"] + (["--model", model] if model else [])
    if settings:
        flags += ["--settings", str(Path(settings).resolve())]
    return flags


def _results(path, start_line):
    sid, last = None, None
    with open(path) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "system" and rec.get("subtype") == "init":
                sid = rec.get("session_id") or sid
            if rec.get("type") == "result":
                last = rec
    return sid, last


def run_agent(workdir, model=None, budget_min=60, settings=None, log=print):
    """Run one headless agent in workdir within a wall-clock budget; return a summary dict.

    If a turn ends with an error of the harness or of the API while budget remains, the same
    session is resumed with a neutral message, up to MAX_ATTEMPTS times."""
    if shutil.which("claude") is None:
        raise RuntimeError("the `claude` command (Claude Code CLI) was not found on PATH")
    workdir = Path(workdir).resolve()
    flags = agent_flags(model, settings)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    stream = workdir / "agent_stream.jsonl"
    stream.write_text("")
    start, deadline = time.time(), time.time() + budget_min * 60
    cmd = ["claude", "-p", PROMPT.format(dir=workdir)] + flags
    status, errors, results, sid = "finished", [], [], None
    for _ in range(MAX_ATTEMPTS):
        n0 = sum(1 for _ in open(stream))
        with open(stream, "a") as out, open(workdir / "agent_stderr.log", "a") as err:
            try:
                subprocess.run(cmd, cwd=workdir, stdout=out, stderr=err, env=env,
                               timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                status = "stopped at budget"
        new_sid, res = _results(stream, n0)
        sid = new_sid or sid
        if res:
            results.append(res)
        if status != "finished" or res is None or not res.get("is_error"):
            break
        errors.append(str(res.get("result", ""))[:200])
        log(f"agent turn ended with an error; resuming ({errors[-1][:80]})")
        if deadline - time.time() < 60 or not sid:
            status = "stopped after error"
            break
        cmd = ["claude", "-p", RESUME.format(error=errors[-1][:120]), "--resume", sid] + flags
    usage = {}
    for r in results:
        for k, v in (r.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + v
    return {"status": status, "minutes": round((time.time() - start) / 60, 1), "model": model,
            "attempts": len(errors) + 1, "errors": errors,
            "num_turns": sum(r.get("num_turns") or 0 for r in results), "usage": usage,
            "cost_usd": round(sum(r.get("total_cost_usd") or 0 for r in results), 4),
            "final_message": (results[-1].get("result") if results else None)}


NETWORK_WORDS = ("curl ", "wget ", "pip install", "pip3 install", "uv pip", "conda install",
                 "http://", "https://", "git clone", "apt ", "brew ")


def audit(workdir):
    """Flags of possible rule violations in the transcript: web tools, network or install
    commands, and file paths outside the task directory. A flag is not proof of a violation;
    read the transcript (agent_stream.jsonl) to judge it."""
    workdir = Path(workdir).resolve()
    flags = []
    with open(workdir / "agent_stream.jsonl") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            for c in rec.get("message", {}).get("content", []):
                if c.get("type") != "tool_use":
                    continue
                name, inp = c.get("name"), c.get("input", {})
                if name in DISALLOWED:
                    flags.append(f"used tool {name}")
                text = json.dumps(inp)
                if name == "Bash":
                    cmd = inp.get("command", "")
                    for w in NETWORK_WORDS:
                        if w in cmd:
                            flags.append(f"command contains '{w.strip()}': {cmd[:120]}")
                            break
                for p in inp.values() if isinstance(inp, dict) else []:
                    if isinstance(p, str) and p.startswith("/") and not p.startswith(str(workdir)):
                        if name in ("Read", "Write", "Edit", "Glob", "Grep"):
                            flags.append(f"{name} outside the task directory: {p[:120]}")
    return flags
