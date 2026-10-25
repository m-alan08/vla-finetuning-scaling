import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FAKES = Path(__file__).resolve().parent / "probe_fake_workers.py"

spec = importlib.util.spec_from_file_location("probe", REPO / "probe" / "probe.py")
probe = importlib.util.module_from_spec(spec)
sys.modules["probe"] = probe
spec.loader.exec_module(probe)


def fake(name, **kw):
    return probe.run_isolated(name, kw.pop("kwargs", None), module_path=FAKES, **kw)


def result(name, status, intentional=False):
    return probe.CheckResult(name, status, "reason", intentional_skip=intentional)


def test_isolated_worker_pass_round_trip():
    res = fake("ok", kwargs={"value": 7})
    assert res["status"] == "PASS"
    assert res["details"]["value"] == 7


def test_exception_in_worker_becomes_fail_not_crash():
    res = fake("raises")
    assert res["status"] == "FAIL"
    assert "RuntimeError" in res["reason"]
    assert "deliberate failure" in res["reason"]


def test_segfault_in_worker_is_caught():
    res = fake("segfault")
    assert res["status"] == "FAIL"
    assert "SIGSEGV" in res["reason"]


def test_timeout_is_enforced():
    start = time.time()
    res = fake("sleeps", kwargs={"seconds": 60}, timeout=2)
    assert res["status"] == "FAIL"
    assert "timed out" in res["reason"]
    assert time.time() - start < 20


def test_worker_that_exits_without_result_is_fail():
    res = fake("exits_silently")
    assert res["status"] == "FAIL"
    assert "code 3" in res["reason"]
    assert "some noise" in res["details"]["stdout_tail"]


def test_invalid_status_from_worker_is_fail():
    res = fake("bad_status")
    assert res["status"] == "FAIL"


def test_unknown_worker_is_fail():
    res = fake("does_not_exist")
    assert res["status"] == "FAIL"


def test_stdin_and_env_reach_worker():
    assert fake("reads_stdin", stdin_text="n\n")["details"]["line"] == "n"
    assert fake("reads_env", env={"PROBE_TEST_VAR": "hello"})["reason"] == "hello"


def test_check_that_raises_in_parent_is_reported_and_run_continues():
    def boom(ctx):
        raise ValueError("parent side bug")

    def fine(ctx):
        return probe.outcome("PASS", "fine")

    checks = [probe.Check("boom", boom), probe.Check("fine", fine)]
    results = probe.run_checks(checks, probe.Context())
    assert [r.status for r in results] == ["FAIL", "PASS"]
    assert "parent side bug" in results[0].reason


def test_check_returning_garbage_status_is_fail():
    checks = [probe.Check("odd", lambda ctx: {"status": "OK", "reason": "x"})]
    assert probe.run_checks(checks, probe.Context())[0].status == "FAIL"


def test_laptop_profile_skips_simulator_checks_intentionally():
    called = []
    checks = [probe.Check("libero_env", lambda ctx: called.append(1) or probe.outcome("PASS", "x"))]
    res = probe.run_checks(checks, probe.Context(laptop=True))[0]
    assert res.status == "SKIP" and res.intentional_skip
    assert called == []


def test_verdict_ready_when_all_pass():
    v = probe.compute_verdict([result("a", "PASS"), result("b", "PASS")], {"a"})
    assert v["verdict"] == "READY" and v["exit_code"] == 0 and v["failed"] == []


def test_verdict_ready_with_intentional_skip():
    v = probe.compute_verdict([result("a", "PASS"), result("b", "SKIP", intentional=True)], {"a"})
    assert v["verdict"] == "READY" and v["exit_code"] == 0


def test_verdict_partial_when_noncritical_fails():
    v = probe.compute_verdict([result("a", "PASS"), result("b", "FAIL")], {"a"})
    assert v["verdict"] == "PARTIAL" and v["exit_code"] == 1 and v["failed"] == ["b"]


def test_verdict_partial_when_check_skipped_unexpectedly():
    v = probe.compute_verdict([result("a", "PASS"), result("b", "SKIP")], {"a"})
    assert v["verdict"] == "PARTIAL" and v["unexpected_skips"] == ["b"]


def test_verdict_not_ready_when_critical_fails():
    v = probe.compute_verdict([result("a", "FAIL"), result("b", "FAIL")], {"a"})
    assert v["verdict"] == "NOT READY" and v["exit_code"] == 2
    assert v["failed"] == ["a", "b"] and v["critical_failed"] == ["a"]
    assert "a, b" in probe.verdict_line(v)


def test_main_writes_report_and_returns_exit_code(tmp_path):
    checks = [
        probe.Check("good", lambda ctx: probe.outcome("PASS", "fine"), critical=True),
        probe.Check("bad", lambda ctx: probe.outcome("FAIL", "broken"), critical=True),
    ]
    out = io.StringIO()
    report = tmp_path / "r.json"
    code = probe.main(["--report", str(report)], checks=checks, stream=out)
    assert code == 2
    data = json.loads(report.read_text())
    assert data["verdict"]["verdict"] == "NOT READY"
    assert data["verdict"]["failed"] == ["bad"]
    assert [c["status"] for c in data["checks"]] == ["PASS", "FAIL"]
    text = out.getvalue()
    assert "[PASS] good" in text and "[FAIL] bad" in text and "VERDICT: NOT READY (failed: bad)" in text


def test_main_only_filter_and_unknown_name(tmp_path):
    checks = [probe.Check("x", lambda ctx: probe.outcome("PASS", "x")), probe.Check("y", lambda ctx: probe.outcome("FAIL", "y"))]
    assert probe.main(["--only", "x", "--report", str(tmp_path / "a.json")], checks=checks, stream=io.StringIO()) == 0
    assert probe.main(["--only", "nope"], checks=checks, stream=io.StringIO()) == 2


def test_disk_check_threshold(tmp_path):
    paths = {"tmp": tmp_path}
    assert probe.check_disk(probe.Context(min_free_gb=0.001), paths)["status"] == "PASS"
    assert probe.check_disk(probe.Context(min_free_gb=1e9), paths)["status"] == "FAIL"


def test_disk_check_measures_nearest_existing_folder(tmp_path):
    missing = tmp_path / "not" / "yet" / "created"
    res = probe.check_disk(probe.Context(min_free_gb=0.001), {"cache": missing})
    assert res["status"] == "PASS"
    assert res["details"]["cache"]["measured_at"] == str(tmp_path.resolve())


def test_import_first_tries_old_then_new_layout():
    module, errors = probe.import_first(["no_such_pkg.layout_a", "json"], "dumps")
    assert module == "json"
    assert "no_such_pkg.layout_a" in errors
    module, errors = probe.import_first(["no_such_pkg.a", "json"], "no_such_attribute")
    assert module is None and len(errors) == 2


def test_blank_image_detection():
    pytest.importorskip("numpy")
    assert probe.is_blank(probe.image_stats([[0, 0], [0, 0]]))
    assert not probe.is_blank(probe.image_stats([[0, 255], [30, 200]]))


def test_cli_end_to_end_exit_code(tmp_path):
    report = tmp_path / "cli.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "probe" / "probe.py"), "--only", "system", "--report", str(report)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERDICT: READY" in proc.stdout
    assert json.loads(report.read_text())["checks"][0]["name"] == "system"


def test_real_train_step_loss_drops_on_this_machine():
    pytest.importorskip("torch")
    res = probe.run_isolated("train_step", {"device": "cpu"}, timeout=180)
    assert res["status"] == "PASS", res
    assert res["details"]["last_loss"] < 0.1 * res["details"]["first_loss"]


def test_require_gpu_fails_without_cuda():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA GPU is present")
    assert probe.run_isolated("torch", {"require_gpu": True}, timeout=120)["status"] == "FAIL"
    assert probe.run_isolated("torch", {"require_gpu": False}, timeout=120)["status"] == "PASS"


def test_find_command_looks_beside_the_python_interpreter(monkeypatch):
    monkeypatch.setenv("PATH", "")
    name = Path(sys.executable).name
    assert probe.find_command(name) == str(Path(sys.executable).parent / name)
    assert probe.find_command("no-such-command-xyz") is None


def test_short_error_keeps_only_the_first_line():
    err = probe.short_error(RuntimeError("first line\nsecond line\nthird"))
    assert err == "RuntimeError: first line"


def test_headless_backends_try_egl_then_osmesa():
    assert probe.headless_backends("Linux") == ("egl", "osmesa")
    assert probe.headless_backends("Darwin") == ("egl", "osmesa", "cgl")
