# Instructions for Claude: running ADR for a user

This repository implements agentic distributional regression (ADR; Tedesco and Finazzi,
arXiv:2610.37611). If a user asks you to run ADR on their data with this repository, follow
the steps below. They are the same as those of the `/adr` skill in `skills/adr/SKILL.md`.

**The estimate must come from a separate, isolated agent started by the `adr` command.** Do
not estimate the model yourself, do not edit the returned `model.py`, and do not describe the
data to the agent: that would no longer be ADR.

1. **Arguments.** Identify the csv file, the response column (`-y`) (or two columns for a bivariate
   response) and, optionally, the covariates (default: all other columns), the model (default
   `claude-opus-5-5`) and the time budget in minutes (default 60). Check the header of the csv;
   if the response is unclear, ask the user.
2. **Environment.** From the repository folder, run `PY="$(scripts/ensure_env.sh)"`: it creates
   once a Python environment with the package (in `~/.adr/venv`) and prints its python. Check
   that the `claude` command exists (`command -v claude`); if not, tell the user that Claude
   Code must be installed.
3. **Start.** Tell the user in one or two lines what will run (response, covariates, model,
   budget, held-out fraction) and that the training data are sent to the Anthropic API. Then
   run, in the background, from the user's working directory:

   ```bash
   "$PY" -m adr fit DATA.csv --response Y [--covariates ...] --budget MIN --model NAME --out adr_runs/<name>
   ```

   Wait for it to finish without polling.
4. **Report.** Read `<out>/report.md` and present the held-out NLL against the two reference
   models, the interval coverage, the returned program with a plain-language reading (family;
   covariates in location, scale and mixture weights; changes of regime), the audit flags and
   the contract status, and the location of the files. Mention that another run may give a
   different program.

Run `python -m pytest tests` after changing the code.
