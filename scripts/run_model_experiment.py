"""Run the new bounded experiment without modifying historical pilot scripts."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "src"))

from readcue.model_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(entrypoint=__file__))
