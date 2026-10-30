"""Tests that need no agent: contract, task preparation, evaluation pipeline and audit.
Run with `python -m pytest tests` from the repository root."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import adr
from adr import agent
from adr.contract import check

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "branched.csv"
PROGRAM = ROOT / "examples" / "branched_program.py"


def test_contract_accepts_example_program():
    X = pd.read_csv(DATA)[["x"]].to_numpy()
    assert check(PROGRAM, 1, 1, 80, xs=X, domain=(X.min(0), X.max(0))) == []


def test_contract_rejects_loops_and_too_many_constants(tmp_path):
    loop = tmp_path / "loop.py"
    loop.write_text("from adr.dist import Normal\n\ndef model(x):\n    s = 0\n"
                    "    for v in x:\n        s = s + v\n    return Normal(s, 1.0)\n")
    assert any("For" in p for p in check(loop, 1, 1, 80))
    many = tmp_path / "many.py"
    terms = " + ".join(f"{k}.5 * x[0]" for k in range(90))
    many.write_text(f"from adr.dist import Normal\n\ndef model(x):\n    return Normal({terms}, 1.0)\n")
    assert any("too many numeric literals" in p for p in check(many, 1, 1, 80))


def test_prepare_task(tmp_path):
    X, y = np.random.default_rng(0).normal(size=(50, 3)), np.random.default_rng(1).normal(size=50)
    d = agent.prepare_task(tmp_path / "task", X, y, domain=np.column_stack([X.min(0), X.max(0)]))
    assert {p.name for p in d.iterdir()} >= {"data.csv", "TASK.md", "adr", "domain.csv"}
    assert list(pd.read_csv(d / "data.csv").columns) == ["x0", "x1", "x2", "y"]
    task = (d / "TASK.md").read_text()
    assert "Covariate domain" in task and "--domain domain.csv" in task
    assert {p.name for p in (d / "adr").iterdir()} == {"__init__.py", "dist.py", "contract.py"}


def test_fit_with_given_program_and_load(tmp_path):
    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "adr", "fit", str(DATA), "-y", "y",
                    "--program", str(PROGRAM), "--out", str(out)], check=True, cwd=ROOT)
    info = json.loads((out / "run.json").read_text())
    ev = info["evaluation"]
    assert info["contract"] == "ok"
    assert ev["nll"] < ev["reference"]["linear Gaussian"] < ev["reference"]["normal, no covariates"]
    assert (out / "report.md").exists() and (out / "density.png").exists()
    fit = adr.load(out)
    df = pd.read_csv(DATA).head(5)
    assert np.all(np.isfinite(fit.logpdf(df)))
    assert fit.sample(df, n=3, seed=0).shape == (5, 3)


def test_audit_flags(tmp_path):
    rec = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "WebFetch", "input": {"url": "https://example.org"}},
        {"type": "tool_use", "name": "Bash", "input": {"command": "pip install torch"}},
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/etc/passwd"}}]}}
    (tmp_path / "agent_stream.jsonl").write_text(json.dumps(rec) + "\n")
    flags = agent.audit(tmp_path)
    assert len(flags) == 3


def test_agent_runs_in_a_neutral_directory(tmp_path, monkeypatch):
    """The agent sees its working directory: it must not contain the file or column names."""
    from adr import cli
    seen = {}

    def fake_run_agent(task, model, budget, settings):
        seen["dir"] = str(task)
        (Path(task) / "model.py").write_text(PROGRAM.read_text())
        (Path(task) / "agent_stream.jsonl").write_text("")
        return {"status": "finished", "minutes": 0.0, "model": model, "num_turns": 1,
                "cost_usd": 0.0}

    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    monkeypatch.setenv("ADR_WS", str(tmp_path / "ws"))
    data = tmp_path / "secret_survey.csv"
    pd.read_csv(DATA).rename(columns={"y": "income"}).to_csv(data, index=False)
    out = tmp_path / "out"
    cli.main(["fit", str(data), "-y", "income", "--out", str(out)])
    assert "secret_survey" not in seen["dir"] and "income" not in seen["dir"]
    assert seen["dir"].startswith(str(tmp_path / "ws"))
    assert (out / "task" / "model.py").exists() and not Path(seen["dir"]).exists()
    assert json.loads((out / "run.json").read_text())["contract"] == "ok"
