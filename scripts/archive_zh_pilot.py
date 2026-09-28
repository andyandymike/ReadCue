"""Compatible archive entry point; historical inference/scoring sources stay frozen."""
import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from readcue.evaluation import archive_pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive_pilot(PROJECT, args.run, args.output)


if __name__ == "__main__":
    main()
