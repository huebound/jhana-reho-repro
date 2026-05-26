from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .afni import validate_against_afni, validate_subjects_against_afni
from .classify import ClassificationConfig, run_benchmark, run_modes
from .datasets import prepare_ds002748
from .reho import compute_reho_dataset


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jhana-repro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare-ds002748", help="Download ds002748 metadata and prepare derivative paths")
    add_common_workdir(prep)
    prep.add_argument("--derivatives-dir", type=Path, default=None)
    prep.add_argument("--clone-derivatives", action="store_true")
    prep.set_defaults(func=cmd_prepare)

    reho = subparsers.add_parser("compute-reho", help="Compute Schaefer ROI ReHo features")
    add_common_workdir(reho)
    reho.add_argument("--derivatives-dir", type=Path, default=None)
    reho.add_argument("--max-subjects", type=int, default=None)
    reho.add_argument("--subjects", nargs="+", default=None)
    reho.add_argument("--output", type=Path, default=None)
    reho.set_defaults(func=cmd_compute_reho)

    classify = subparsers.add_parser("classify", help="Run public-data classifiers")
    classify.add_argument("--reho", type=Path, required=True)
    classify.add_argument("--output-dir", type=Path, required=True)
    classify.add_argument("--mode", choices=["leak-safe", "compat", "both"], default="both")
    classify.add_argument("--n-permutations", type=int, default=1000)
    classify.add_argument("--compat-target-per-subject", type=int, default=8)
    classify.set_defaults(func=cmd_classify)

    benchmark = subparsers.add_parser("benchmark-public", help="Run repeated-seed public classifier benchmark")
    benchmark.add_argument("--reho", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--mode", choices=["leak-safe", "compat", "both"], default="both")
    benchmark.add_argument("--seeds", type=int, default=50)
    benchmark.add_argument("--seed-start", type=int, default=0)
    benchmark.add_argument("--compat-target-per-subject", type=int, default=8)
    benchmark.set_defaults(func=cmd_benchmark)

    run = subparsers.add_parser("run-public", help="Prepare metadata, compute ReHo, and run classifiers")
    add_common_workdir(run)
    run.add_argument("--derivatives-dir", type=Path, default=None)
    run.add_argument("--clone-derivatives", action="store_true")
    run.add_argument("--max-subjects", type=int, default=None)
    run.add_argument("--subjects", nargs="+", default=None)
    run.add_argument("--mode", choices=["leak-safe", "compat", "both"], default="both")
    run.add_argument("--n-permutations", type=int, default=1000)
    run.set_defaults(func=cmd_run_public)

    validate = subparsers.add_parser("validate-reho-afni", help="Compare Python ReHo with AFNI 3dReHo")
    add_common_workdir(validate)
    validate.add_argument("--derivatives-dir", type=Path, default=None)
    validate.add_argument("--subject", default="sub-01")
    validate.add_argument("--subjects", nargs="+", default=None)
    validate.add_argument("--output", type=Path, default=None)
    validate.set_defaults(func=cmd_validate_afni)

    inspect = subparsers.add_parser("inspect-mrp", help="Write the known notebook contract document")
    inspect.add_argument("--out", type=Path, default=Path("docs/notebook_contract.md"))
    inspect.set_defaults(func=cmd_inspect_mrp)
    return parser


def add_common_workdir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path, required=True)


def cmd_prepare(args: argparse.Namespace) -> None:
    info = prepare_ds002748(args.work_dir, args.derivatives_dir, args.clone_derivatives)
    print(json.dumps(info, indent=2))


def cmd_compute_reho(args: argparse.Namespace) -> None:
    output = args.output or (args.work_dir / "Data" / "1ReHo" / "reho_output.csv")
    derivatives = args.derivatives_dir or (args.work_dir / "ds002748-fmriprep")
    df = compute_reho_dataset(args.work_dir, derivatives, output, args.max_subjects, args.subjects)
    print(f"Wrote {len(df)} ReHo rows to {output}")


def cmd_classify(args: argparse.Namespace) -> None:
    config = ClassificationConfig(
        n_permutations=args.n_permutations,
        compat_target_per_subject=args.compat_target_per_subject,
    )
    results = run_modes(args.reho, args.output_dir, args.mode, config)
    print(json.dumps({mode: result["ensemble"] for mode, result in results.items()}, indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    config = ClassificationConfig(
        n_permutations=0,
        compat_target_per_subject=args.compat_target_per_subject,
    )
    frames = run_benchmark(
        reho_csv=args.reho,
        output_dir=args.output_dir,
        seeds=args.seeds,
        seed_start=args.seed_start,
        mode=args.mode,
        config=config,
    )
    print(frames["summary"].to_json(orient="records", indent=2))


def cmd_run_public(args: argparse.Namespace) -> None:
    info = prepare_ds002748(args.work_dir, args.derivatives_dir, args.clone_derivatives)
    derivatives = Path(info["derivatives_dir"])
    reho_csv = args.work_dir / "Data" / "1ReHo" / "reho_output.csv"
    should_compute_reho = not reho_csv.exists() or args.subjects is not None or args.max_subjects is not None
    if should_compute_reho:
        compute_reho_dataset(args.work_dir, derivatives, reho_csv, args.max_subjects, args.subjects)
    config = ClassificationConfig(n_permutations=args.n_permutations)
    results = run_modes(reho_csv, args.work_dir / "classification", args.mode, config)
    print(json.dumps({mode: result["ensemble"] for mode, result in results.items()}, indent=2))


def cmd_validate_afni(args: argparse.Namespace) -> None:
    derivatives = args.derivatives_dir or (args.work_dir / "ds002748-fmriprep")
    if args.subjects:
        result = validate_subjects_against_afni(args.work_dir, derivatives, args.subjects, args.output)
    else:
        result = validate_against_afni(args.work_dir, derivatives, args.subject)
    print(json.dumps(result, indent=2))


def cmd_inspect_mrp(args: argparse.Namespace) -> None:
    from importlib import resources

    text = resources.files("jhana_repro").joinpath("notebook_contract_template.md").read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
