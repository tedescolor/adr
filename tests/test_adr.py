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


# ---------------------------------------------------------------- mixed and discrete responses
from adr.score import Support, analyse_response, loglik  # noqa: E402
from adr.dist import Mixture, PointMass, LogNormal, Poisson  # noqa: E402

EX = ROOT / "examples"


def test_inspect_suggests_the_response_type():
    y = lambda f: pd.read_csv(EX / f)["y"].to_numpy(float)
    assert analyse_response(y("branched.csv"))["type"] == "continuous"
    r = analyse_response(y("zero_inflated.csv"))
    assert r["type"] == "mixed" and r["atoms"] == [0.0]
    r = analyse_response(y("counts.csv"))
    assert r["type"] == "discrete" and r["support"] == "naturals"
    likert = 1 + np.random.default_rng(0).binomial(4, 0.4, 500).astype(float)
    r = analyse_response(likert)
    assert r["type"] == "discrete" and "--support 1:5" in " ".join(r["notes"])


def test_support_parsing():
    assert Support.parse("naturals").contains([0, 3, 7]).all()
    assert not Support.parse("naturals").contains([-1, 0.5]).any()
    s = Support.parse("1:5")
    assert s.finite and s.contains([1, 5]).all() and not s.contains([0, 6]).any()
    assert Support.parse("0:inf:0.5").contains([0, 0.5, 10.5]).all()
    assert Support.parse("0,1,2,5").contains([5]).all() and not Support.parse("0,1,2,5").contains([3]).any()


def test_mixed_and_discrete_scores():
    d = Mixture([0.3, 0.7], [PointMass(0.0), LogNormal(0.0, 1.0)])
    ll = loglik(d, [0.0, 1.0], "mixed", [0.0])
    assert np.isclose(ll[0], np.log(0.3)) and np.isclose(ll[1], np.log(0.7) + LogNormal(0, 1).logpdf(1.0))
    assert np.isclose(loglik(Poisson(2.0), [3], "discrete", support=Support.parse("naturals"))[0],
                      np.log(np.exp(-2) * 8 / 6))
    assert np.isneginf(loglik(Poisson(2.0), [2.5], "discrete", support=Support.parse("naturals"))[0])


def test_contract_for_mixed_and_discrete(tmp_path):
    X = pd.read_csv(EX / "zero_inflated.csv")[["x"]].to_numpy()
    yz = pd.read_csv(EX / "zero_inflated.csv")["y"].to_numpy()
    ok = check(EX / "zero_inflated_program.py", 1, 1, 80, xs=X, mode="mixed", atoms=[0.0], ys=yz)
    assert ok == []
    wrong_atom = check(EX / "zero_inflated_program.py", 1, 1, 80, xs=X, mode="mixed", atoms=[1.0], ys=yz)
    assert wrong_atom  # mass at 0, which is not declared
    cont = tmp_path / "cont.py"
    cont.write_text("from adr.dist import Normal\n\ndef model(x):\n    return Normal(x[0], 1.0)\n")
    Xc = pd.read_csv(EX / "counts.csv")[["x1", "x2"]].to_numpy()
    yc = pd.read_csv(EX / "counts.csv")["y"].to_numpy()
    assert check(EX / "counts_program.py", 2, 1, 80, xs=Xc, mode="discrete",
                 support=Support.parse("naturals"), ys=yc) == []
    assert check(cont, 2, 1, 80, xs=Xc, mode="discrete", support=Support.parse("naturals"), ys=yc)


def test_continuous_task_uses_the_library_of_the_paper(tmp_path):
    X, y = np.random.default_rng(0).normal(size=(30, 1)), np.random.default_rng(1).normal(size=30)
    d = agent.prepare_task(tmp_path / "c", X, y)
    for f in ("dist.py", "contract.py"):
        assert (d / "adr" / f).read_bytes() == (ROOT / "adr" / "_paper" / f).read_bytes()
    d = agent.prepare_task(tmp_path / "m", X, np.abs(y) * (y > 0), mode="mixed", atoms=[0.0])
    assert (d / "adr" / "task.py").exists() and "point masses" in (d / "TASK.md").read_text()


def test_fit_mixed_and_discrete_with_given_programs(tmp_path):
    for name, flag, refs in [("zero_inflated", "--mixed", "logistic atoms + linear Gaussian"),
                             ("counts", "--discrete", "negative binomial regression")]:
        out = tmp_path / name
        subprocess.run([sys.executable, "-m", "adr", "fit", str(EX / f"{name}.csv"), "-y", "y", flag,
                        "--program", str(EX / f"{name}_program.py"), "--out", str(out)],
                       check=True, cwd=ROOT)
        info = json.loads((out / "run.json").read_text())
        assert info["contract"] == "ok" and np.isfinite(info["evaluation"]["nll"])
        assert info["evaluation"]["nll"] <= info["evaluation"]["reference"][refs] + 0.05


def test_printing_and_plotting(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "adr", "fit", str(DATA), "-y", "y",
                    "--program", str(PROGRAM), "--out", str(out)], check=True, cwd=ROOT)
    fit = adr.load(out)
    assert "response type: continuous" in repr(fit) and "x[0] = x" in repr(fit)
    assert repr(fit.distribution({"x": 0.8})).startswith("Mixture(weights=[0.5, 0.5]; Normal(")
    q = fit.quantile({"x": 0.8}, [0.1, 0.5, 0.9])
    assert np.all(np.diff(q) > 0)
    fit.plot([{"x": 0.3}, {"x": 0.8}])


def test_example_run_loads_and_matches_its_report():
    fit = adr.load(EX / "example_run")
    df = pd.read_csv(DATA)
    assert np.all(np.isfinite(fit.logpdf(df.head(50))))
    assert abs(fit.info["evaluation"]["nll"] - 0.2616) < 1e-3
