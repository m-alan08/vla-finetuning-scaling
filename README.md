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
| 0 | Repo skeleton | Done (committed locally, not yet pushed) |
| 1 | Environment probe on the cluster | Not started |
| 2 | Design document | Not started (stub only) |
| 3a | Pipeline and smoke tests | Not started |
| 3b | One full run of the smallest configuration | Not started |
| 3c | Full experiment grid | Not started |

No code has been written and no experiment has been run. There are no results yet.

Dataset revision: not yet pinned (will be recorded here before any training run).

## Layout

| Path | Contents |
|---|---|
| `docs/PROJECT_BRIEF.md` | The question, experiment plan and compute notes |
| `docs/DESIGN.md` | Plain-language design of the pipeline (stub until Stage 2) |
| `probe/` | Environment probe, its SLURM launcher and its tests |
| `src/` | Project Python package |
| `configs/` | One configuration file per experiment setting |
| `slurm/` | sbatch scripts for the HCE cluster |
| `tests/` | Unit and smoke tests for Mac and cluster |
| `results/` | Small result files only (JSON, CSV, PNG) |
| `CLAUDE.md` | Working rules for AI-assisted sessions on this repo |

## Quick start

Only the repository itself exists so far:

```bash
git clone https://github.com/m-alan08/vla-finetuning-scaling.git
cd vla-finetuning-scaling
```

Environment setup, dataset download, training, evaluation and plotting commands will be added here as each stage is built and verified.

## License

MIT, see [LICENSE](LICENSE).
