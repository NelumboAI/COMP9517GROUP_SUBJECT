"""Launch the unchanged traditional baseline in an isolated run directory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the run directory and start training. Default is dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run-name may contain only letters, numbers, '_' and '-'")

    workspace = Path(__file__).resolve().parents[2]
    working_root = workspace / "reproduction" / "working_code"
    train_script = working_root / "src" / "traditional" / "train.py"
    config_relative = Path("configs") / "baseline_local.yaml"
    config_path = working_root / config_relative
    run_dir = workspace / "reproduction" / "outputs" / "traditional" / args.run_name

    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {run_dir}")

    command = [sys.executable, str(train_script), config_relative.as_posix()]
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Unchanged training script: {train_script}")
    print(f"Local config: {config_path}")
    print(f"New run directory: {run_dir}")
    print(f"Command: {subprocess.list2cmdline(command)}")

    if not args.execute:
        print("Nothing was created. Add --execute to start the full baseline.")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = run_dir / "run_metadata.json"
    log_path = run_dir / "run.log"
    metadata = {
        "run_name": args.run_name,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "training_script": str(train_script),
        "config": str(config_path),
        "command": command,
    }
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(working_root) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()

    metadata["status"] = "completed" if return_code == 0 else "failed"
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["return_code"] = return_code
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
