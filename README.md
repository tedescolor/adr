# ADR: agentic distributional regression

ADR estimates the **conditional distribution of a response given covariates** with an LLM
agent. An isolated Claude agent examines your data, chooses and fits a model, and returns it as
a short Python program: for every value of the covariates, the program returns a probability
distribution (a normal, a Student-*t*, a mixture, a bivariate factorization, ...) whose
parameters are explicit formulas in the covariates. You get a density, quantiles and random
draws for every covariate value, and a formula you can read.

The method and its statistical properties are described in

> Lorenzo Tedesco and Francesco Finazzi (2026). *Agentic Distributional Regression: Conditional
> Density Estimation through Agent-Written Programs*. [arXiv:2610.37611](https://arxiv.org/abs/2610.37611).

The code and results of the paper are in
[ADR_replication](https://github.com/tedescolor/ADR_replication). This repository is the tool
to use the method on your own data.

![How ADR works: the agent receives the data, the library, the output contract, the scoring criterion and a time budget; it proposes, fits, checks and revises models, and returns a short program whose numbers are the parameters and whose formulas are the model; if the program satisfies the contract it defines the estimated conditional density.](docs/adr_overview.png)

## Quick start in Claude Code: `/adr`

Install the plugin once, inside Claude Code:

```
/plugin marketplace add tedescolor/adr
/plugin install adr@adr
```

Then, in any project:

```
/adr:adr data.csv -y income                      # response `income`; all other columns are covariates
/adr:adr data.csv -x age education -y income     # covariates `age` and `education`; response `income`
/adr:adr data.csv -y income --budget 30 --model claude-sonnet-5-5
```

`-y` (or `--response`) gives the response, two columns for a bivariate response, and `-x` (or
`--covariates`) the covariates; without `-x`, all other columns are covariates. A short form
without flags also works: the last column name is the response and the names before it are the
covariates (`/adr:adr data.csv age education income`). You can also write it in words:
`/adr:adr data.csv predict income from age and education`.
Claude prepares the task, starts the isolated agent in the background, and, when it has
finished, shows you the held-out evaluation and explains the returned program.

## Or: give this repository to Claude

Without installing anything, you can ask Claude Code something like:

> Use https://github.com/tedescolor/adr to run ADR on `data.csv` with response `income`, with
> Claude Opus 5.5 and a 30-minute budget.

Claude clones the repository and follows [CLAUDE.md](CLAUDE.md).

## Command line

```bash
pip install git+https://github.com/tedescolor/adr
adr fit data.csv --response income [--covariates age education] --budget 60 --model claude-opus-5-5
```

| option | meaning (default) |
|---|---|
| `--response`, `-y` | response column, or two columns for a bivariate response |
| `--covariates`, `-x` | covariate columns (all other columns) |
| `--model` | Claude model of the agent (`claude-opus-5-5`) |
| `--budget` | wall-clock time budget of the agent in minutes (60) |
| `--test-fraction` / `--test FILE` | rows held out for evaluation (0.2) / a separate test file |
| `--keep-names` | tell the agent the column names (by default the columns are anonymous, as in the paper) |
| `--sandbox` | run the agent's shell in the Claude Code sandbox, without network (needs bubblewrap on Linux) |
| `--no-domain` | do not require a valid distribution on the whole box of observed covariate values |
| `--out` | output directory (`adr_runs/<data>-<time>`) |
| `--dry-run` | prepare the task without starting the agent |
| `--program FILE` | evaluate an existing `model.py` instead of running an agent |

`adr check model.py --data data.csv -y income` checks a program against the output contract, and
`adr report RUN_DIR` rewrites a report.

## What you get

In the output directory:

- `report.md`: held-out negative log-likelihood of ADR next to two simple references (a
  normal distribution without covariates and a linear Gaussian model), coverage of the central
  50% and 90% intervals, the program, the agent's notes and the figures;
- `model.py`: the returned program; `x[j]` is the j-th covariate, in the order listed in the
  report;
- `notes.md`: the agent's account of what it tried and why it chose the program;
- `calibration.png` (PIT histogram), `density.png` (one covariate) or `joint.png` (bivariate
  response);
- `task/`: everything the agent saw and did, including the full transcript
  (`agent_stream.jsonl`).

## Using the estimate in Python

```python
import adr, numpy as np
fit = adr.load("adr_runs/data-income")
d = fit.distribution({"age": 40, "education": 16})   # an adr.dist.Distribution
d.logpdf(np.array([30000.0])); d.cdf(np.array([30000.0])); d.ppf(0.9)
d.sample(np.random.default_rng(0), 1000)
fit.logpdf(test_df)          # log-density of each row of a data frame
```

## How it works

1. The training rows are written to a new task directory with a random name (in `/var/tmp/adr`,
   or `$ADR_WS`), with anonymous column names (`x0`, `x1`, ..., `y`), together with the distribution library `adr/dist.py`, the contract checker
   `adr/contract.py` and a fixed task description (`TASK.md`). By default the agent also
   receives the range of each covariate over the whole dataset and must return a valid
   distribution on that box. After the run, the directory is moved to `<out>/task`.
2. A headless Claude Code session (`claude -p`) is started in that directory with a fixed
   prompt, restricted tools (no web tools, file tools confined to the directory) and the time
   budget. The agent explores the data, proposes and fits models, validates them as it sees
   fit, and writes `model.py` and `notes.md`. It usually stops before the budget expires.
3. The program is checked against the **output contract**: imports only from `adr.dist` and
   `math`, at most 80 numerical constants and 500 syntax-tree nodes, no loops, and a valid
   distribution at training rows and on the covariate box.
4. The program is evaluated on the held-out rows, and the transcript is scanned for possible
   rule violations (web or network use, files outside the task directory).

## Things to know

- **Your data leave your computer.** The training rows are sent to the Anthropic API through
  Claude Code. Do not use ADR on data you are not allowed to share with it.
- **Cost and time.** In the paper, a run with Claude Opus 5.5 cost about 1 USD in median
  (at most about 3.5 USD) and took 13 to 18 minutes in median; larger datasets and longer
  budgets can cost more. The cost of each run is shown in its report.
- **Variability.** The agent is random: another run on the same data may return a different
  program. For important analyses, run it more than once and compare the held-out scores.
- **The LLM matters.** In the paper, Claude Opus 5.5 was clearly more accurate than Claude
  Sonnet 5.5, which often stopped early with a simpler model.
- **Interpretation.** A short formula is easy to read, but it is still a fitted model: check
  it outside the observed covariate range before relying on it there.
- **Requirements.** Python 3.10 or later, and Claude Code (the `claude` command) logged in or
  with an `ANTHROPIC_API_KEY`. Responses must be continuous; covariates must be numeric (encode
  categories as 0/1 indicators).

## Example

`examples/branched.csv` contains 1,000 draws from the univariate design of the paper, whose
conditional law changes from one to two branches at x = 0.5. To check the installation without
calling an agent:

```bash
adr fit examples/branched.csv -y y --program examples/branched_program.py --out adr_runs/example
```

and, to run the agent on it, drop `--program`.

## Citation

```bibtex
@misc{tedesco2026adr,
  author        = {Tedesco, Lorenzo and Finazzi, Francesco},
  title         = {Agentic Distributional Regression: Conditional Density Estimation through
                   Agent-Written Programs},
  year          = {2026},
  eprint        = {2610.37611},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME}
}
```

## License

MIT (see [LICENSE](LICENSE)).
