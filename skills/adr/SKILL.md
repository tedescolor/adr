---
name: adr
description: Run agentic distributional regression (ADR) on a csv file. An isolated Claude agent estimates the conditional distribution of a response given covariates and returns a short, readable program; the run is evaluated on held-out rows. Use when the user asks to run ADR, or to estimate a conditional density or distributional regression with this plugin.
argument-hint: <data.csv> [-x covariate ...] -y <response> [--budget MIN] [--model NAME]
allowed-tools: Bash, Read
---

# Run ADR

The user invoked ADR with: `$ARGUMENTS`

ADR is the method of Tedesco and Finazzi, *Agentic Distributional Regression*
(arXiv:2610.37611). The estimate must come from a **separate, isolated agent** started by the
`adr` command, which sees only the anonymized training data, the distribution library and a
fixed task description. **Do not estimate the model yourself, do not edit `model.py`, and do
not tell the agent anything about the data**: that would no longer be ADR.

## 1. Read the arguments

The first argument is the csv file (header row required). The response and the covariates can
be given in three ways:

1. **Explicitly**, as on the command line: `-y RESPONSE` (or `--response`; two names for a
   bivariate response) and, optionally, `-x COV1 COV2 ...` (or `--covariates`; default: all
   other columns). Example: `/adr:adr data.csv -x age education -y income`.
2. **By position**: the names after the file are columns, and **the last name is the response**;
   the names before it are the covariates. With a single name, it is the response and all
   other columns are covariates. Example: `/adr:adr data.csv age education income`.
3. **In words**, for example `/adr:adr data.csv predict income from age and education`: then
   identify the response and the covariates from the sentence and the csv header.

Options passed through unchanged: `--budget MIN` (default 60), `--model NAME` (default
`claude-opus-5-5`), `--test-fraction F` (default 0.2), `--test FILE`, `--keep-names`,
`--sandbox`, `--seed N`, `--no-domain`, the size limits `--max-constants K` (default 80)
and `--max-nodes N` (default 500), and the response-type options `--mixed`, `--atoms A ...`,
`--discrete`, `--support SPEC`.

Read the csv header (`head -1`) to check that the columns exist. Before starting, state which
column is the response and which are the covariates. If this is ambiguous, show the header
and ask the user.

## 2. Prepare the environment

```bash
PY="$("${CLAUDE_PLUGIN_ROOT}/scripts/ensure_env.sh")"
command -v claude >/dev/null || echo "MISSING: the claude CLI is needed to start the agent"
```

The first call creates a Python environment with the `adr` package (a minute or two).
If `claude` is missing, tell the user to install Claude Code (the `claude` command) and stop.

## 3. Inspect the data and choose the response type

```bash
"$PY" -m adr inspect DATA.csv --response RESPONSE [--covariates C1 C2 ...]
```

Summarize its findings for the user: the covariates (flag non-numeric ones, which must be
encoded or left out), the response, and the suggested response type:

- **continuous** (the setting of the paper): no extra option;
- **mixed**, a continuous response with point masses (for example many exact zeros): add
  `--mixed`; the atoms are those reported (values in at least 1% of the rows) unless the user
  gave `--atoms`;
- **discrete**: add `--discrete --support SPEC`. If the suggested support has no upper bound but
  the response might have a maximum (a score from 1 to 5, successes out of m trials, ...), ask
  the user whether there is a maximum before starting, and pass a finite support (`a:b`).

If the user already gave a response-type option, keep it, and mention it if `inspect` suggests
something else. Mixed and discrete responses are an extension beyond the paper, which studies
continuous responses.

## 4. Start the run

Tell the user, in one or two lines, what will run: the response, the covariates, the model,
the time budget, the held-out fraction, and that the training data are sent to the Anthropic
API through Claude Code (in the paper, a run with Claude Opus 5.5 cost about 1 USD in
median and at most about 3.5 USD, and took 13 to 18 minutes in median). Then start it **in the background** (it can take up to the time budget):

```bash
"$PY" -m adr fit DATA.csv --response RESPONSE [--covariates C1 C2 ...] [TYPE OPTIONS] \
      --budget MIN --model NAME --out adr_runs/NAME_OF_RUN
```

Choose `adr_runs/<csv stem>-<response>` as output directory (add a suffix if it exists). Do
not poll: wait for the background command to finish.

## 5. Report

When the command has finished, read `<out>/report.md` and give the user:

- the response type (with the atoms or the support), the held-out NLL of ADR next to the two
  reference models, and the interval coverage;
- the returned program, with a plain-language reading of it: which distribution family,
  which covariates enter the location, the scale or the mixture weights, any change of regime
  (in `model.py`, `x[j]` is the j-th covariate, as listed in the report);
- the audit flags, if any, and the status of the output contract;
- where the files are (`report.md`, `model.py`, `notes.md`, figures), and how to use the
  estimate from Python (`adr.load(<out>)`, shown at the end of the report).

If the contract is not satisfied or no program was written, say so plainly and show the end
of `<out>/task/agent_stderr.log`. Mention that another run may return a different program.
