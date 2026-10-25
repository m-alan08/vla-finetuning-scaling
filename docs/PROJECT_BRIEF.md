# Project brief

## Title

How much does pretraining help VLA fine-tuning, and what should you freeze?

## Question

When fine-tuning a pretrained vision-language-action (VLA) model on a small number of robot demos, how does success rate scale with the number of demos, and how does it depend on which parts of the model are trainable?

This follows an X series on RT-2 (does web knowledge transfer to control), pi0 (a VLM plus a flow-matching action expert) and ACT (imitation learning from scratch on low-cost hardware).

## Benchmark

LIBERO simulation, suites `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`.

## Models

| Model | Checkpoint | Role |
|---|---|---|
| SmolVLA | `lerobot/smolvla_base` | Main workhorse, full fine-tuning |
| pi0 | `lerobot/pi0_base` | Full fine-tuning, only if memory allows |
| ACT | trained from scratch | The no-pretraining baseline |

## Experiments

1. **Data scaling.** Train each model on roughly 10%, 25%, 50% and 100% of the available demos per task. Report success rate versus demos per task. Confirm how many demos per task the dataset actually has before fixing the fractions.
2. **Freeze ablation** on SmolVLA (and pi0 if feasible), three settings:
   - everything trainable
   - everything except the vision encoder
   - action head only

   Report trainable parameter counts for each setting.
3. **Forgetting.** Evaluate on tasks or suites not trained on, before and after fine-tuning.
4. **Seeds.** 3 seeds for every reported number, with mean and spread. Pin the dataset revision and record it in the README.

## Deliverables

- A results table
- A plot of success rate versus demos per task (one line per model and setup)
- The freeze ablation table, with trainable parameter counts
- A README a stranger can use to reproduce everything

## Compute and environment

- **Cluster:** HCE at Georgia Tech. GPU nodes are x86 with 4 GPUs each: H200 141GB, H100 80GB, L40S, L40, L4. Use the `general` partition.
- **Queueing:** class work has priority on these nodes. Keep jobs short, checkpoint often, make every job resumable.
- **Login node:** never run heavy work there. Use `sbatch` for anything needing a GPU. If a partition or GPU name is rejected, run `sinfo` and fix the script.
- **pi0 memory:** openpi's docs list over 70 GB per GPU for pi0 full fine-tuning, so pi0 needs an H100 or H200, or sharding across the 4 GPUs of one node. Check what LeRobot's pi0 training actually needs before committing.
- **LeRobot and LIBERO (starting point, verify against the installed version):** install with `pip install -e ".[libero]"` from a LeRobot clone, set `MUJOCO_GL=egl` on headless machines, use dataset `lerobot/libero` (needs `torchcodec` or `av`), train with `lerobot-train`, evaluate with `lerobot-eval`.
- **Laptop:** MacBook Pro M5 with 16 GB. Everything except simulator rendering and real training must also run there (CPU or MPS).

## Stages

| Stage | Content | Gate |
|---|---|---|
| 0 | Repo skeleton | Committed locally |
| 1 | Environment probe (`probe/probe.py`, unit tests, `probe/run_probe.sbatch`), run on the cluster until READY | Verdict READY |
| 2 | `docs/DESIGN.md` in plain language: scripts, experiment grid, result computation, risks, GPU-hour estimates | Owner's go-ahead before anything larger than a smoke test |
| 3 | Build the pipeline, smoke tests, one full run of the smallest configuration, then the grid | Short report after each stage |

### Stage 1 probe checks

Each check prints PASS, FAIL or SKIP with a reason. Risky checks run in their own subprocess so a hard crash becomes a FAIL line.

- **system:** host, GPUs, SLURM info
- **disk:** at least 20 GB free in the working folder and the Hugging Face cache
- **torch:** imports and sees the GPU (an option makes this required)
- **train_step:** a tiny network trains on the device and its loss really drops, plus a bf16 test on GPU
- **lerobot:** installed, and SmolVLA code imports (report pi0 but do not require it; try both old and new module layouts)
- **video_decode:** `torchcodec` or `av` installed
- **mujoco_headless:** renders an image with no screen, trying EGL then OSMesa, and the image must not be blank
- **libero_import:** imports and lists the suites
- **libero_env:** one LIBERO task is created, reset and stepped once
- **hub:** `lerobot/smolvla_base`, `lerobot/pi0_base` and the dataset `lerobot/libero` resolve on Hugging Face from this machine

The probe ends with a verdict (READY, PARTIAL, or NOT READY with the failed names) and saves a JSON report. Its unit tests check its own logic: failures are reported rather than crashing, a segfaulting subprocess is caught, timeouts work, and the verdict and exit code are right. The SLURM script uses the general partition, 1 GPU, 8 CPUs, 32G, 30 minutes.

## Verification requirements

The owner understands the concepts and math of robot learning but does not read the code line by line. Correctness must be checkable without reading code, through tests whose claims are easy to reason about:

- a data split has the intended number of demos per task
- frozen parts get no gradient updates
- trainable parameter counts match the freeze setting
- result aggregation gives the expected mean and spread on made-up numbers
- a resumed job continues from its checkpoint

These tests run on both the Mac (CPU or MPS) and the cluster GPUs.
