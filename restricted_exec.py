"""Run a subprocess with bounded capture and publish only sanitized status."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

MAX_TIMEOUT_SECONDS = 30 * 60
MAX_CAPTURE_BYTES = 1_048_576


@dataclass
class ExecutionResult:
    child_exit_code: int | None
    timed_out: bool
    output_limit_exceeded: bool
    stdout: bytes
    stderr: bytes

    def public_status(self) -> dict:
        return {
            "schema": "restricted-exec-status/1",
            "child_exit_code": self.child_exit_code,
            "timed_out": self.timed_out,
            "output_limit_exceeded": self.output_limit_exceeded,
            "stdout_bytes": len(self.stdout),
            "stderr_bytes": len(self.stderr),
            "stdout_sha256": hashlib.sha256(self.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(self.stderr).hexdigest(),
        }


def run_bounded(command: list[str], *, timeout_seconds: float, max_output_bytes: int) -> ExecutionResult:
    if not command:
        raise ValueError("command is required")
    if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be in (0, {MAX_TIMEOUT_SECONDS}]")
    if not 1024 <= max_output_bytes <= MAX_CAPTURE_BYTES:
        raise ValueError(f"max_output_bytes must be between 1024 and {MAX_CAPTURE_BYTES}")

    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"},
    )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    exceeded = threading.Event()

    def drain(name: str, stream) -> None:
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                buf = buffers[name]
                remaining = max_output_bytes - len(buf)
                if remaining > 0:
                    buf.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    exceeded.set()
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    return
        finally:
            stream.close()

    assert proc.stdout is not None and proc.stderr is not None
    threads = [
        threading.Thread(target=drain, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", proc.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    for thread in threads:
        thread.join(timeout=2)

    return ExecutionResult(
        child_exit_code=proc.returncode,
        timed_out=timed_out,
        output_limit_exceeded=exceeded.is_set(),
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command with bounded, non-echoed output capture.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-output", type=int, default=65536, help="Maximum bytes captured per stream.")
    parser.add_argument("--stdout-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command

    result = run_bounded(command, timeout_seconds=args.timeout, max_output_bytes=args.max_output)
    status = result.public_status()
    status["timeout_seconds"] = args.timeout
    status["max_output_bytes_per_stream"] = args.max_output
    _atomic_write(args.status_file, (json.dumps(status, sort_keys=True) + "\n").encode())

    success = (
        result.child_exit_code == 0
        and not result.timed_out
        and not result.output_limit_exceeded
    )
    if success:
        _atomic_write(args.stdout_file, result.stdout)
        return 0
    args.stdout_file.unlink(missing_ok=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
