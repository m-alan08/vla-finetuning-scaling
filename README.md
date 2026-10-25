# How much does pretraining help VLA fine-tuning, and what should you freeze?

When you fine-tune a pretrained vision-language-action (VLA) model on a small number of robot demos, how does success rate scale with the number of demos, and how does it depend on which parts of the model are trainable?

This project measures that on the LIBERO simulation benchmark (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) with three models:

- **SmolVLA** (`lerobot/smolvla_base`), the main workhorse, fully fine-tuned
- **pi0** (`lerobot/pi0_base`), fully fine-tuned if GPU memory allows
- **ACT** trained from scratch, the no-pretraining baseline

Experiments: data scaling (about 10%, 25%, 50% and 100% of demos per task), a freeze ablation (everything trainable, vision encoder frozen, action head only), forgetting on tasks not trained on, and 3 seeds for every reported number. The full plan is in [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md).

## Status

| Stage | What | State |
|---|---|---|
| 0 | Repo skeleton | Done (pushed to GitHub) |
| 1 | Environment probe on the cluster | In progress: probe written, unit tests pass, full probe READY on the Mac; not yet run on the cluster |
| 2 | Design document | Not started (stub only) |
| 3a | Pipeline and smoke tests | Not started |
| 3b | One full run of the smallest configuration | Not started |
| 3c | Full experiment grid | Not started |

No experiment has been run. There are no results yet.

Dataset revision: not yet pinned (will be recorded here before any training run). The probe reports the current revision of `lerobot/libero`.

Software: LeRobot v0.6.1 (commit `7e241bd`), Python 3.12, installed by `setup_env.sh`.

## Layout

| Path | Contents |
|---|---|
| `docs/PROJECT_BRIEF.md` | The question, experiment plan and compute notes |
| `docs/DESIGN.md` | Plain-language design of the pipeline (stub until Stage 2) |
| `probe/` | Environment probe, its SLURM launcher and its tests |
| `src/` | Project Python package |
| `configs/` | One configuration file per experiment setting |
| `slurm/` | sbatch scripts for the HCE cluster, and `env.sh` (shared environment variables) |
| `setup_env.sh` | Creates the Python environment from nothing |
| `tests/` | Unit and smoke tests for Mac and cluster |
| `results/` | Small result files only (JSON, CSV, PNG) |
| `CLAUDE.md` | Working rules for AI-assisted sessions on this repo |

## Quick start

```bash
git clone https://github.com/m-alan08/vla-finetuning-scaling.git
cd vla-finetuning-scaling
bash setup_env.sh
```

`setup_env.sh` installs uv if needed, clones LeRobot v0.6.1 into `third_party/`, creates `.venv` with Python 3.12 and installs LeRobot with the LIBERO, SmolVLA and pi0 extras. Caches (Hugging Face, uv, LIBERO config) go to `.cache/` inside the repo. To put them elsewhere, for example on a scratch filesystem, set `VLA_CACHE=/path/to/scratch` before running it and before submitting jobs. If the cluster's NVIDIA driver is too old for the default PyTorch build, set `TORCH_INDEX_URL` (for example `https://download.pytorch.org/whl/cu126`) and rerun.

Check the environment on the cluster (GPU job, about 5 to 10 minutes):

```bash
sbatch probe/run_probe.sbatch
```

Check it on a Mac (simulator checks skipped):

```bash
source slurm/env.sh && python probe/probe.py --laptop
```

Run the unit tests:

```bash
source slurm/env.sh && python -m pytest tests -q
```

See [probe/README.md](probe/README.md) for what each check and verdict means.

## License

MIT, see [LICENSE](LICENSE).
