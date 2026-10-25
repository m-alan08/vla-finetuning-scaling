import argparse
import dataclasses
import datetime
import importlib
import importlib.util
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

READY = "READY"
PARTIAL = "PARTIAL"
NOT_READY = "NOT READY"

EXIT_CODES = {READY: 0, PARTIAL: 1, NOT_READY: 2}

RESULT_MARKER = "__PROBE_RESULT__"

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_FILE = Path(__file__).resolve()

LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

SMOLVLA_CANDIDATES = [
    "lerobot.policies.smolvla.modeling_smolvla",
    "lerobot.common.policies.smolvla.modeling_smolvla",
]
PI0_CANDIDATES = [
    "lerobot.policies.pi0.modeling_pi0",
    "lerobot.common.policies.pi0.modeling_pi0",
]
ACT_CANDIDATES = [
    "lerobot.policies.act.modeling_act",
    "lerobot.common.policies.act.modeling_act",
]

HUB_REQUIRED = [("model", "lerobot/smolvla_base"), ("dataset", "lerobot/libero")]
HUB_OPTIONAL = [("model", "lerobot/pi0_base")]

CRITICAL_CHECKS = {"disk", "torch", "train_step", "lerobot", "video_decode", "hub"}
LAPTOP_SKIPS = {"mujoco_headless", "libero_import", "libero_env"}


@dataclasses.dataclass
class CheckResult:
    name: str
    status: str
    reason: str
    details: dict = dataclasses.field(default_factory=dict)
    seconds: float = 0.0
    intentional_skip: bool = False


@dataclasses.dataclass
class Check:
    name: str
    func: object
    critical: bool = False


@dataclasses.dataclass
class Context:
    require_gpu: bool = False
    laptop: bool = False
    min_free_gb: float = 20.0
    timeout_scale: float = 1.0
    device: str = "auto"
    results: dict = dataclasses.field(default_factory=dict)


def outcome(status, reason, **details):
    return {"status": status, "reason": reason, "details": details}


def signal_name(returncode):
    try:
        return signal.Signals(-returncode).name
    except (ValueError, AttributeError):
        return f"signal {-returncode}"


def tail(text, lines=15):
    parts = (text or "").strip().splitlines()
    return "\n".join(parts[-lines:])


def run_isolated(worker, kwargs=None, timeout=60.0, env=None, stdin_text=None, module_path=None):
    cmd = [sys.executable, str(PROBE_FILE), "--worker", worker, "--worker-kwargs", json.dumps(kwargs or {})]
    if module_path is not None:
        cmd += ["--worker-module", str(module_path)]
    full_env = dict(os.environ)
    full_env.update(env or {})
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_text if stdin_text is not None else "",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return outcome(FAIL, f"timed out after {timeout:.0f}s", stdout_tail=tail(out), stderr_tail=tail(err))
    elapsed = time.time() - start
    if proc.returncode < 0:
        return outcome(
            FAIL,
            f"subprocess crashed with {signal_name(proc.returncode)}",
            returncode=proc.returncode,
            stderr_tail=tail(proc.stderr),
            seconds=round(elapsed, 2),
        )
    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                payload = None
            break
    if payload is None:
        return outcome(
            FAIL,
            f"subprocess exited with code {proc.returncode} without reporting a result",
            returncode=proc.returncode,
            stdout_tail=tail(proc.stdout),
            stderr_tail=tail(proc.stderr),
        )
    if payload.get("status") not in (PASS, FAIL, SKIP):
        return outcome(FAIL, f"subprocess reported an invalid status {payload.get('status')!r}")
    payload.setdefault("details", {})
    return payload


def load_worker(name, module_path=None):
    if module_path is None:
        module = sys.modules[__name__]
    else:
        spec = importlib.util.spec_from_file_location("probe_worker_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    func = getattr(module, f"worker_{name}", None)
    if func is None:
        raise SystemExit(f"unknown worker {name}")
    return func


def worker_main(name, kwargs_json, module_path=None):
    try:
        func = load_worker(name, module_path)
        result = func(**json.loads(kwargs_json))
    except BaseException as exc:
        result = outcome(
            FAIL,
            f"{type(exc).__name__}: {exc}",
            traceback=tail(traceback.format_exc(), 25),
        )
    sys.stdout.flush()
    print(RESULT_MARKER + json.dumps(result, default=str), flush=True)
    return 0


def import_first(candidates, attribute):
    errors = {}
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except BaseException as exc:
            errors[module_name] = short_error(exc)
            continue
        if hasattr(module, attribute):
            return module_name, errors
        errors[module_name] = f"module has no attribute {attribute}"
    return None, errors


def find_command(name):
    found = shutil.which(name)
    if found:
        return found
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.exists() else None


def short_error(exc, limit=400):
    text = f"{type(exc).__name__}: {exc}"
    first = text.strip().splitlines()[0] if text.strip() else text
    return first[:limit]


def pick_device(requested):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def worker_torch(require_gpu=False):
    import torch

    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    gpus = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": props.name, "memory_gb": round(props.total_memory / 1e9, 1)})
    details = {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "mps_available": mps,
        "gpus": gpus,
    }
    if cuda:
        return outcome(PASS, f"torch {torch.__version__} sees {len(gpus)} GPU(s): {gpus[0]['name']}", **details)
    if require_gpu:
        return outcome(FAIL, f"torch {torch.__version__} imported but no CUDA GPU is visible and a GPU is required", **details)
    where = "MPS" if mps else "CPU only"
    return outcome(PASS, f"torch {torch.__version__} imported, no CUDA GPU ({where}), GPU not required", **details)


def worker_train_step(device="auto", steps=200, seed=0):
    import torch

    dev = pick_device(device)
    torch.manual_seed(seed)
    x = torch.randn(256, 16)
    true_w = torch.randn(16, 1)
    y = x @ true_w + 0.01 * torch.randn(256, 1)
    x, y = x.to(dev), y.to(dev)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = losses[0], losses[-1]
    details = {"device": dev, "first_loss": first, "last_loss": last, "steps": steps}
    if not (last == last) or last >= 0.1 * first:
        return outcome(FAIL, f"loss did not drop enough on {dev}: {first:.4f} -> {last:.4f}", **details)
    msg = f"loss dropped on {dev}: {first:.4f} -> {last:.4f}"
    if dev != "cuda":
        details["bf16"] = "not tested (needs a CUDA GPU)"
        return outcome(PASS, msg + "; bf16 not tested without a CUDA GPU", **details)
    if not torch.cuda.is_bf16_supported():
        details["bf16"] = "unsupported"
        return outcome(FAIL, msg + "; but this GPU does not support bf16", **details)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)
        bf16_loss = torch.nn.functional.mse_loss(out.float(), y)
    bf16_loss.backward()
    grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters())
    details.update({"bf16_output_dtype": str(out.dtype), "bf16_loss": bf16_loss.item(), "bf16_grads_finite": grads_finite})
    if out.dtype != torch.bfloat16 or not grads_finite:
        return outcome(FAIL, msg + f"; bf16 test failed (dtype {out.dtype}, finite grads {grads_finite})", **details)
    return outcome(PASS, msg + "; bf16 autocast forward and backward OK", **details)


def worker_lerobot():
    import lerobot

    version = getattr(lerobot, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version("lerobot")
        except Exception:
            version = "unknown"
    smolvla, smolvla_err = import_first(SMOLVLA_CANDIDATES, "SmolVLAPolicy")
    act, act_err = import_first(ACT_CANDIDATES, "ACTPolicy")
    pi0, pi0_err = import_first(PI0_CANDIDATES, "PI0Policy")
    scripts = {name: find_command(name) for name in ("lerobot-train", "lerobot-eval")}
    details = {
        "version": version,
        "smolvla_module": smolvla,
        "act_module": act,
        "pi0_module": pi0,
        "import_errors": {"smolvla": smolvla_err, "act": act_err, "pi0": pi0_err},
        "scripts": scripts,
    }
    problems = []
    if smolvla is None:
        problems.append("SmolVLA does not import")
    if act is None:
        problems.append("ACT does not import")
    missing_scripts = [k for k, v in scripts.items() if v is None]
    if missing_scripts:
        problems.append("missing commands " + ", ".join(missing_scripts))
    pi0_note = "pi0 imports" if pi0 else "pi0 does not import (not required)"
    if problems:
        return outcome(FAIL, f"lerobot {version}: " + "; ".join(problems) + f"; {pi0_note}", **details)
    layout = "new layout" if smolvla.startswith("lerobot.policies") else "old layout"
    return outcome(PASS, f"lerobot {version} ({layout}): SmolVLA and ACT import; {pi0_note}", **details)


def worker_video_decode():
    found = {}
    errors = {}
    try:
        import torchcodec
        from torchcodec.decoders import VideoDecoder

        found["torchcodec"] = getattr(torchcodec, "__version__", "unknown")
    except BaseException as exc:
        errors["torchcodec"] = short_error(exc)
    try:
        import av

        found["av"] = getattr(av, "__version__", "unknown")
    except BaseException as exc:
        errors["av"] = short_error(exc)
    if not found:
        return outcome(FAIL, "neither torchcodec nor av can be imported", errors=errors)
    names = ", ".join(f"{k} {v}" for k, v in found.items())
    unavailable = "".join(f"; {k} unavailable ({v})" for k, v in errors.items())
    return outcome(PASS, f"video decoding available: {names}{unavailable}", found=found, errors=errors)


MUJOCO_XML = """
<mujoco>
  <visual><global offwidth="64" offheight="64"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.2 0.3 0.8 1"/>
    <body pos="0 0 0.3"><geom type="box" size="0.2 0.2 0.2" rgba="0.9 0.2 0.2 1"/></body>
    <camera name="cam" pos="0 -1.5 1" xyaxes="1 0 0 0 0.6 0.8"/>
  </worldbody>
</mujoco>
"""


def image_stats(pixels):
    import numpy as np

    arr = np.asarray(pixels, dtype=np.float64)
    return {"shape": list(arr.shape), "mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def is_blank(stats):
    return stats["std"] < 1.0 or stats["max"] - stats["min"] < 5.0


def worker_mujoco_render():
    import mujoco

    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    renderer.update_scene(data, camera="cam")
    pixels = renderer.render()
    stats = image_stats(pixels)
    renderer.close()
    stats["mujoco_version"] = mujoco.__version__
    stats["backend"] = os.environ.get("MUJOCO_GL")
    if is_blank(stats):
        return outcome(FAIL, f"rendered image is blank (std {stats['std']:.2f})", **stats)
    return outcome(PASS, f"rendered a non-blank 64x64 image (std {stats['std']:.1f})", **stats)


def libero_stdin():
    return "n\n"


def worker_libero_import():
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    available = sorted(suites.keys())
    missing = [s for s in LIBERO_SUITES if s not in suites]
    counts = {}
    for name in LIBERO_SUITES:
        if name in suites:
            counts[name] = suites[name]().n_tasks
    if missing:
        return outcome(FAIL, f"missing suites: {', '.join(missing)}", available=available, task_counts=counts)
    summary = ", ".join(f"{k} ({v} tasks)" for k, v in counts.items())
    return outcome(PASS, f"LIBERO imports, suites: {summary}", available=available, task_counts=counts)


def worker_libero_env(suite="libero_spatial", task_id=0, size=128):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    try:
        env.seed(0)
        env.reset()
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
    finally:
        env.close()
    image_keys = sorted(k for k in obs if k.endswith("_image"))
    if not image_keys:
        return outcome(FAIL, "step returned no camera images", obs_keys=sorted(obs))
    stats = image_stats(obs[image_keys[0]])
    details = {"suite": suite, "task": task.language, "image_keys": image_keys, "image": stats, "reward": float(reward), "backend": os.environ.get("MUJOCO_GL")}
    if is_blank(stats):
        return outcome(FAIL, f"camera image {image_keys[0]} is blank", **details)
    return outcome(PASS, f"created, reset and stepped '{task.language}' ({suite} task {task_id}); image not blank", **details)


def hub_request(path, token=None, timeout=20):
    req = urllib.request.Request(f"https://huggingface.co/{path}", headers={"User-Agent": "vla-probe"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def hub_lookup(kind, repo_id, token=None):
    prefix = "api/models" if kind == "model" else "api/datasets"
    try:
        info = hub_request(f"{prefix}/{repo_id}", token)
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "sha": info.get("sha"), "gated": info.get("gated"), "last_modified": info.get("lastModified")}


def worker_hub():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    repos = {}
    for kind, repo_id in HUB_REQUIRED + HUB_OPTIONAL:
        repos[repo_id] = hub_lookup(kind, repo_id, token)
    dataset_meta = None
    if repos["lerobot/libero"]["ok"]:
        sha = repos["lerobot/libero"]["sha"]
        try:
            meta = hub_request(f"datasets/lerobot/libero/resolve/{sha}/meta/info.json", token)
            dataset_meta = {k: meta.get(k) for k in ("codebase_version", "total_episodes", "total_tasks", "total_frames", "fps")}
        except Exception as exc:
            dataset_meta = {"error": f"{type(exc).__name__}: {exc}"}
    details = {"repos": repos, "dataset_meta": dataset_meta, "token_present": bool(token)}
    failed = [r for _, r in HUB_REQUIRED if not repos[r]["ok"]]
    pi0_ok = repos["lerobot/pi0_base"]["ok"]
    pi0_note = "pi0_base resolves" if pi0_ok else f"pi0_base does not resolve ({repos['lerobot/pi0_base']['error']}, not required)"
    if failed:
        errs = "; ".join(f"{r}: {repos[r]['error']}" for r in failed)
        return outcome(FAIL, f"could not resolve {errs}; {pi0_note}", **details)
    extra = ""
    if dataset_meta and "total_episodes" in dataset_meta:
        extra = f"; dataset has {dataset_meta['total_episodes']} episodes over {dataset_meta['total_tasks']} tasks"
    return outcome(PASS, f"smolvla_base and lerobot/libero resolve; {pi0_note}{extra}", **details)


def nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as exc:
        return [f"nvidia-smi failed: {exc}"]
    return [line.strip() for line in out.stdout.splitlines() if line.strip()] or [tail(out.stderr, 3)]


def check_system(ctx):
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in {
        "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS", "SLURM_CLUSTER_NAME",
    }}
    gpus = nvidia_smi()
    details = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "slurm": slurm,
        "nvidia_smi": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    where = f"SLURM job {slurm['SLURM_JOB_ID']} on {slurm.get('SLURM_JOB_PARTITION', '?')}" if "SLURM_JOB_ID" in slurm else "not inside a SLURM job"
    gpu_text = "; ".join(gpus) if gpus else "no nvidia-smi"
    return outcome(PASS, f"{details['host']}, {details['machine']}, Python {details['python']}, {where}, GPUs: {gpu_text}", **details)


def hf_cache_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def existing_parent(path):
    path = Path(path).expanduser().resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_gb(path):
    return shutil.disk_usage(existing_parent(path)).free / 1e9


def check_disk(ctx, paths=None):
    paths = paths or {"working folder": Path.cwd(), "Hugging Face cache": hf_cache_dir()}
    details = {}
    low = []
    for label, path in paths.items():
        gb = free_gb(path)
        details[label] = {"path": str(path), "measured_at": str(existing_parent(path)), "free_gb": round(gb, 1)}
        if gb < ctx.min_free_gb:
            low.append(f"{label} has {gb:.1f} GB free")
    text = ", ".join(f"{k} {v['free_gb']} GB" for k, v in details.items())
    if low:
        return outcome(FAIL, f"need at least {ctx.min_free_gb:.0f} GB: " + "; ".join(low), **details)
    return outcome(PASS, f"free space OK (>= {ctx.min_free_gb:.0f} GB): {text}", **details)


def scaled(ctx, seconds):
    return seconds * ctx.timeout_scale


def check_torch(ctx):
    return run_isolated("torch", {"require_gpu": ctx.require_gpu}, timeout=scaled(ctx, 120))


def check_train_step(ctx):
    if ctx.results.get("torch", {}).get("status") == FAIL and not ctx.results["torch"].get("details", {}).get("torch_version"):
        return outcome(SKIP, "torch did not import")
    return run_isolated("train_step", {"device": ctx.device}, timeout=scaled(ctx, 180))


def check_lerobot(ctx):
    return run_isolated("lerobot", timeout=scaled(ctx, 300))


def check_video_decode(ctx):
    return run_isolated("video_decode", timeout=scaled(ctx, 120))


def headless_backends(system=None):
    system = system or platform.system()
    return ("egl", "osmesa", "cgl") if system == "Darwin" else ("egl", "osmesa")


def check_mujoco_headless(ctx):
    attempts = {}
    for backend in headless_backends():
        env = {"MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}
        res = run_isolated("mujoco_render", env=env, timeout=scaled(ctx, 120))
        attempts[backend] = {"status": res["status"], "reason": res["reason"]}
        if res["status"] == PASS:
            res["details"]["attempts"] = attempts
            res["details"]["working_backend"] = backend
            res["reason"] = f"{backend}: {res['reason']}"
            return res
    reasons = "; ".join(f"{k}: {v['reason']}" for k, v in attempts.items())
    return outcome(FAIL, f"no headless backend works ({reasons})", attempts=attempts)


def render_env(ctx):
    mj = ctx.results.get("mujoco_headless", {})
    backend = mj.get("details", {}).get("working_backend") or os.environ.get("MUJOCO_GL") or "egl"
    return {"MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend}


def check_libero_import(ctx):
    return run_isolated("libero_import", env=render_env(ctx), stdin_text=libero_stdin(), timeout=scaled(ctx, 300))


def check_libero_env(ctx):
    if ctx.results.get("libero_import", {}).get("status") != PASS:
        return outcome(SKIP, "depends on libero_import, which did not pass")
    if ctx.results.get("mujoco_headless", {}).get("status") != PASS:
        return outcome(SKIP, "depends on mujoco_headless, which did not pass")
    return run_isolated("libero_env", env=render_env(ctx), stdin_text=libero_stdin(), timeout=scaled(ctx, 600))


def check_hub(ctx):
    return run_isolated("hub", timeout=scaled(ctx, 120))


def default_checks():
    funcs = [
        ("system", check_system),
        ("disk", check_disk),
        ("torch", check_torch),
        ("train_step", check_train_step),
        ("lerobot", check_lerobot),
        ("video_decode", check_video_decode),
        ("mujoco_headless", check_mujoco_headless),
        ("libero_import", check_libero_import),
        ("libero_env", check_libero_env),
        ("hub", check_hub),
    ]
    return [Check(name, func, name in CRITICAL_CHECKS) for name, func in funcs]


def run_one(check, ctx):
    start = time.time()
    if ctx.laptop and check.name in LAPTOP_SKIPS:
        return CheckResult(check.name, SKIP, "skipped on the laptop profile (simulator rendering runs on the cluster)", intentional_skip=True)
    try:
        res = check.func(ctx)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        res = outcome(FAIL, f"probe error: {type(exc).__name__}: {exc}", traceback=tail(traceback.format_exc(), 25))
    status = res.get("status")
    if status not in (PASS, FAIL, SKIP):
        res = outcome(FAIL, f"check returned invalid status {status!r}")
    return CheckResult(check.name, res["status"], res["reason"], res.get("details", {}), round(time.time() - start, 2))


def run_checks(checks, ctx, stream=None):
    results = []
    for check in checks:
        res = run_one(check, ctx)
        ctx.results[check.name] = dataclasses.asdict(res)
        results.append(res)
        if stream is not None:
            print(f"[{res.status:4}] {res.name:16} {res.reason}", file=stream, flush=True)
    return results


def compute_verdict(results, critical):
    failed = [r.name for r in results if r.status == FAIL]
    unexpected_skips = [r.name for r in results if r.status == SKIP and not r.intentional_skip]
    critical_failed = [n for n in failed if n in critical]
    if critical_failed:
        verdict = NOT_READY
    elif failed or unexpected_skips:
        verdict = PARTIAL
    else:
        verdict = READY
    return {
        "verdict": verdict,
        "exit_code": EXIT_CODES[verdict],
        "failed": failed,
        "critical_failed": critical_failed,
        "unexpected_skips": unexpected_skips,
    }


def verdict_line(v):
    if v["verdict"] == READY:
        return "VERDICT: READY"
    parts = []
    if v["failed"]:
        parts.append("failed: " + ", ".join(v["failed"]))
    if v["unexpected_skips"]:
        parts.append("skipped: " + ", ".join(v["unexpected_skips"]))
    return f"VERDICT: {v['verdict']} ({'; '.join(parts)})"


def write_report(path, results, verdict, ctx, argv):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "argv": argv,
        "options": {"require_gpu": ctx.require_gpu, "laptop": ctx.laptop, "min_free_gb": ctx.min_free_gb, "device": ctx.device},
        "verdict": verdict,
        "checks": [dataclasses.asdict(r) for r in results],
    }
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def default_report_path(report_dir):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    job = os.environ.get("SLURM_JOB_ID")
    name = f"probe_{socket.gethostname().split('.')[0]}_{stamp}" + (f"_job{job}" if job else "") + ".json"
    return Path(report_dir) / name


def parse_args(argv):
    p = argparse.ArgumentParser(description="Environment probe for the VLA fine-tuning project.")
    p.add_argument("--require-gpu", action="store_true", help="fail the torch check if no CUDA GPU is visible")
    p.add_argument("--laptop", action="store_true", help="skip simulator checks that only run on the cluster")
    p.add_argument("--min-free-gb", type=float, default=20.0)
    p.add_argument("--device", default="auto", help="device for train_step: auto, cuda, mps or cpu")
    p.add_argument("--only", default=None, help="comma separated check names to run")
    p.add_argument("--timeout-scale", type=float, default=1.0)
    p.add_argument("--report", default=None, help="path of the JSON report")
    p.add_argument("--report-dir", default=str(REPO_ROOT / "probe" / "reports"))
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-kwargs", default="{}", help=argparse.SUPPRESS)
    p.add_argument("--worker-module", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None, checks=None, stream=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.worker:
        return worker_main(args.worker, args.worker_kwargs, args.worker_module)
    stream = stream or sys.stdout
    ctx = Context(
        require_gpu=args.require_gpu,
        laptop=args.laptop,
        min_free_gb=args.min_free_gb,
        timeout_scale=args.timeout_scale,
        device=args.device,
    )
    checks = checks if checks is not None else default_checks()
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        unknown = wanted - {c.name for c in checks}
        if unknown:
            print(f"unknown check names: {', '.join(sorted(unknown))}", file=stream)
            return 2
        checks = [c for c in checks if c.name in wanted]
    print(f"VLA environment probe on {socket.gethostname()} ({len(checks)} checks)", file=stream, flush=True)
    results = run_checks(checks, ctx, stream)
    verdict = compute_verdict(results, {c.name for c in checks if c.critical})
    report_path = Path(args.report) if args.report else default_report_path(args.report_dir)
    write_report(report_path, results, verdict, ctx, argv)
    print(verdict_line(verdict), file=stream)
    print(f"report: {report_path}", file=stream, flush=True)
    return verdict["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
