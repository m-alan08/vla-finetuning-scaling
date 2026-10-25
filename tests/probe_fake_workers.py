import ctypes
import os
import signal
import sys
import time


def worker_ok(value=1):
    return {"status": "PASS", "reason": f"ok {value}", "details": {"value": value}}


def worker_raises():
    raise RuntimeError("deliberate failure inside the worker")


def worker_segfault():
    if hasattr(signal, "SIGSEGV"):
        os.kill(os.getpid(), signal.SIGSEGV)
    ctypes.string_at(0)


def worker_sleeps(seconds=60):
    time.sleep(seconds)
    return {"status": "PASS", "reason": "woke up", "details": {}}


def worker_exits_silently():
    print("some noise on stdout")
    sys.stdout.flush()
    os._exit(3)


def worker_bad_status():
    return {"status": "MAYBE", "reason": "not a valid status", "details": {}}


def worker_reads_stdin():
    line = sys.stdin.readline().strip()
    return {"status": "PASS", "reason": f"read {line!r}", "details": {"line": line}}


def worker_reads_env():
    return {"status": "PASS", "reason": os.environ.get("PROBE_TEST_VAR", ""), "details": {}}
