"""Core CLI; model libraries are used only by an explicitly requested child run."""
import argparse
import json
from pathlib import Path
import sys

from .artifacts import BASELINE_REPOS, fetch_model, verify_pilot, write_json
from .baselines import run_baseline
from .evaluation import BASELINES, archive_pilot, score_files


def main(argv=None):
    parser = argparse.ArgumentParser(prog="readcue", description="Verify, reproduce and inspect ReadCue experiments.")
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="ReadCue checkout (default: current directory)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify all historical pilot source/data hashes without loading models")
    fetch = commands.add_parser("fetch", help="Fetch a public baseline with upstream checksums")
    fetch.add_argument("--baseline", choices=BASELINE_REPOS, required=True)
    fetch.add_argument("--revision", help="HF commit/ref; omitted only for initial resolution of main")
    fetch.add_argument("--destination", type=Path, required=True)
    run = commands.add_parser("run", help="Invoke the frozen v1 baseline with durable failure records")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--baseline", choices=BASELINES, required=True)
    run.add_argument("--output", type=Path, required=True, help="New directory for predictions, logs and run.json")
    run.add_argument("--model-dir", type=Path)
    run.add_argument("--revision")
    run.add_argument("--device", default="cuda")
    score = commands.add_parser("score", help="Use the unchanged v1 scorer on saved predictions")
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    archive = commands.add_parser("archive", help="Validate and archive all four frozen-pilot baselines")
    archive.add_argument("--run", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            freeze = verify_pilot(args.project)
            result = {"status": "verified", "files": len(freeze["files_sha256"]), "experiment": freeze["version"]}
        elif args.command == "fetch":
            manifest = fetch_model(args.baseline, args.destination, args.revision)
            result = {"repo": manifest["repo"], "revision": manifest["revision"], "files": len(manifest["files"])}
        elif args.command == "run":
            result = run_baseline(args.project, args.inputs, args.baseline, args.output,
                                  model_dir=args.model_dir, revision=args.revision, device=args.device)
        elif args.command == "score":
            if args.output.exists():
                raise FileExistsError("Choose a new score output; preserve previous results")
            result = score_files(args.project, args.cases, args.predictions)
            write_json(args.output, result)
            result = result["by_track"]
        else:
            result = archive_pilot(args.project, args.run, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"readcue: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
