"""Run the analytics module in the required order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run(script_name: str) -> None:
    subprocess.run([sys.executable, str(BASE_DIR / script_name)], check=True)


if __name__ == "__main__":
    run("01_eda.py")
    run("02_modeling.py")
    print("Module 2 completed successfully.")
