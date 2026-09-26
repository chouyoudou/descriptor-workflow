"""Explicit native launcher for a private exported task; not a security sandbox."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--python", default=sys.executable)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    task = root / "task"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GITHUB_", "ACTIONS_", "RUNNER_"))
           and not k.endswith(("_TOKEN", "_SECRET"))}
    started = time.monotonic()
    status = {"producer": "local_native", "exit_code": None,
              "origin": json.loads((root / "EXPORT.json").read_text())["origin"],
              "local_edits_audited": False, "python_command": args.python}
    with tempfile.TemporaryDirectory(prefix="local-task-logs-") as temp:
        stdout_path, stderr_path = Path(temp) / "stdout", Path(temp) / "stderr"
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.run([args.python, "-B", str(task / "run_task.py"),
                                          "--output", str(output)], cwd=task, env=env,
                                         stdout=stdout, stderr=stderr)
            status["exit_code"] = process.returncode
        except BaseException as exc:
            status["launcher_error"] = type(exc).__name__
            raise
        finally:
            status["elapsed_seconds"] = time.monotonic() - started
            for name, path in (("execution.stdout.log", stdout_path), ("execution.stderr.log", stderr_path)):
                if path.exists():
                    import shutil
                    shutil.copyfile(path, output / name)
            (output / "execution.json").write_text(json.dumps(status, indent=2) + "\n")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
