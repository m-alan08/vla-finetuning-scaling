# probe

Environment probe: a script that tries each risky setup step in miniature (GPU, MuJoCo rendering, LIBERO, LeRobot, Hugging Face access), its SLURM launcher, and its unit tests.

## Files

| File | What it is |
|---|---|
| `probe.py` | The probe. Needs only the Python standard library to start, so it can diagnose a broken environment. |
| `run_probe.sbatch` | SLURM job: general partition, 1 GPU, 8 CPUs, 32G, 30 minutes. Runs the probe with the GPU required. Avoids GPU nodes the x86 CUDA environment cannot use: ARM GH200, AMD MI210 and MI300A, and the older P40 and V100 (dgx1) nodes without native bf16. |
| `reports/` | JSON reports, one per run (ignored by git). |
| `../tests/test_probe.py` | Unit tests of the probe's own logic. |

## How to run

On the cluster, from the repository root, after `bash setup_env.sh`:

```bash
sbatch probe/run_probe.sbatch
```

The output goes to `slurm-probe-<jobid>.out` in the repository root.

On the Mac:

```bash
source slurm/env.sh && python probe/probe.py --laptop
```

Unit tests (Mac or cluster, no GPU needed):

```bash
python -m pytest tests/test_probe.py -q
```

## What each check means

Each check prints one line: PASS, FAIL or SKIP, then a reason.

| Check | PASS means | Runs in its own process |
|---|---|---|
| system | Host, Python, SLURM job and GPU list were collected (information only) | no |
| disk | At least 20 GB free in the working folder and in the Hugging Face cache | no |
| torch | PyTorch imports. With `--require-gpu`, a CUDA GPU must also be visible | yes |
| train_step | A tiny network trained for 200 steps and its loss fell below 10% of the starting value. On a CUDA GPU, a bf16 forward and backward pass also ran with finite gradients | yes |
| lerobot | LeRobot imports, SmolVLA and ACT policy code import (new and old module layouts are both tried), and `lerobot-train` and `lerobot-eval` exist. pi0 is reported but not required | yes |
| video_decode | torchcodec or av imports | yes |
| mujoco_headless | MuJoCo rendered a 64x64 test scene without a screen and the image is not blank. Tries EGL, then OSMesa (and CGL on macOS) | yes, one per backend |
| libero_import | LIBERO imports and the four suites are listed with their task counts | yes |
| libero_env | One libero_spatial task was created, reset and stepped once, and its camera image is not blank | yes |
| hub | `lerobot/smolvla_base` and the dataset `lerobot/libero` resolve on Hugging Face. `lerobot/pi0_base` is reported but not required. Also records the dataset revision and its episode and task counts | yes |

"Runs in its own process" means a crash (for example a segmentation fault inside a graphics driver) or a hang only turns that check into a FAIL line; the probe itself keeps going.

## Verdict and exit code

| Verdict | Rule | Exit code |
|---|---|---|
| READY | Every check passed, apart from checks skipped on purpose by `--laptop` | 0 |
| PARTIAL | No critical check failed, but some other check failed or was skipped because something it depends on did not pass | 1 |
| NOT READY | A critical check failed. Critical checks are disk, torch, train_step, lerobot, video_decode and hub | 2 |

The verdict line names every failed or skipped check. The JSON report holds every line plus details (versions, GPU names, error messages, the dataset revision).

## What the unit tests prove

- A check whose worker raises an error becomes a FAIL line with the error message; the probe does not crash.
- A worker that segfaults becomes a FAIL line that names SIGSEGV.
- A worker that hangs is stopped at its timeout and reported as FAIL.
- A worker that exits without reporting, or reports a nonsense status, is a FAIL.
- A bug in the probe itself (outside a worker) becomes a FAIL for that check and the remaining checks still run.
- The verdict and exit code follow the table above for made-up results.
- The JSON report is written and matches the printed lines.
- On a machine without a CUDA GPU, `--require-gpu` turns the torch check into a FAIL.
- The real train_step check passes on the machine running the tests.
