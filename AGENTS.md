# Instructions for Codex sessions

Read `docs/PROJECT_BRIEF.md` first. It holds the research question, the experiment plan, the compute notes and the stage gates. `docs/DESIGN.md` holds the approved design once Stage 2 is done.

## Rules

- Do not put comments in code.
- Do not use em dashes anywhere, including READMEs, docs and commit messages.
- Design for verification without reading code. Include unit tests and smoke tests that run on both the owner's Mac (CPU or MPS) and the cluster GPUs. Tests should check things the owner can reason about: a data split has the intended number of demos per task, frozen parts get no gradient updates, trainable parameter counts match the freeze setting, result aggregation gives the expected mean and spread on made-up numbers, and a resumed job continues from its checkpoint.
- Include every system piece needed to run: environment setup, dataset download, SLURM scripts, checkpointing, evaluation and plotting. Assume nothing is installed.
- When a step fails, report the real error and what was tried. Never mark something as working that was not run.
- At the end of every stage, state plainly what was verified (and how) and what was not verified.
- Keep commits small. Stop and ask before any decision that would cost more than a few GPU-hours, delete files, or change the experiment design.
- Write results to files in the repo (JSON and CSV), not only to the terminal, so plots can be regenerated from them.
- Never commit tokens, keys, datasets, model weights or checkpoints.

## Cluster rules

- Never run heavy work on the login node. Use `sbatch` for anything needing a GPU.
- Use the `general` partition. If a partition or GPU name is rejected, run `sinfo` and fix the script.
- Class work has priority on the GPU nodes: keep jobs short, checkpoint often, make every job resumable.

## Keeping the README honest

Update the status table in `README.md` whenever a stage changes state. A stage is "done" only if it was actually run and its checks passed.
