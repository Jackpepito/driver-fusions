"""Utilities for pipeline logging and subprocess execution."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence


def setup_logger(log_path: Path, name: str = "driver_fusions_pipeline") -> logging.Logger:
    """Create a console + file logger for long-running experiments."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_command(
    cmd: Sequence[str],
    logger: logging.Logger,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> str:
    """Run a command, stream output to logger, and return combined stdout/stderr."""
    cmd_str = " ".join(shlex.quote(token) for token in cmd)
    logger.info("Running command: %s", cmd_str)

    if dry_run:
        logger.info("Dry run enabled, command not executed.")
        return ""

    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    start = time.time()
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        output_lines.append(line)
        logger.info("[subprocess] %s", line)

    return_code = proc.wait()
    elapsed = time.time() - start

    if return_code != 0:
        raise subprocess.CalledProcessError(
            returncode=return_code,
            cmd=list(cmd),
            output="\n".join(output_lines),
        )

    logger.info("Command completed in %.1f seconds", elapsed)
    return "\n".join(output_lines)


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
