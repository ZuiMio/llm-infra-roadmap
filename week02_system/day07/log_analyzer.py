"""Analyze training logs from the command line."""

import argparse
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

COUNT_KEYS = ("total_lines", "error_count", "warning_count")


def analyze_lines(lines):
    """Count total lines and lines containing ERROR or WARNING."""
    # TODO 1:
    # - Count all lines, including blank lines.
    # - Match ERROR and WARNING case-insensitively.
    # - Count each matching line once per category.
    # - Return a dictionary using COUNT_KEYS.
    # raise NotImplementedError("Implement analyze_lines")
    total_lines = 0
    error_count = 0
    warning_count = 0

    for line in lines:
        total_lines += 1
        line_folded = line.casefold()

        if "error" in line_folded:
            error_count += 1

        if "warning" in line_folded:
            warning_count += 1

    return dict(zip(COUNT_KEYS, (total_lines, error_count, warning_count)))



def analyze_file(path):
    """Read a UTF-8 log file and return its counters."""
    # TODO 2:
    # - Convert path to pathlib.Path.
    # - Open it for reading with encoding="utf-8".
    # - Pass the file iterator to analyze_lines().
    # - Let OSError propagate to the caller.
    # raise NotImplementedError("Implement analyze_file")
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        return analyze_lines(file)


def aggregate_reports(reports):
    """Sum the counters from a list of per-file reports."""
    # TODO 3:
    # - Sum each key in COUNT_KEYS.
    # - Return zeros for an empty list.
    # - Do not include the per-file "file" field in the result.
    # raise NotImplementedError("Implement aggregate_reports")
    if reports == []:
        return dict(zip(COUNT_KEYS, (0, 0, 0)))
    total_lines = 0
    error_count = 0
    warning_count = 0
    for per_file in reports:
        total_lines += per_file['total_lines']
        error_count += per_file['error_count']
        warning_count += per_file['warning_count']
    return dict(zip(COUNT_KEYS, (total_lines, error_count, warning_count)))


def write_json_report(reports, total, output_path):
    """Write per-file and aggregate results to a JSON file."""
    # TODO 4:
    # - Convert output_path to pathlib.Path.
    # - Create its parent directory if necessary.
    # - Write {"files": reports, "total": total} as UTF-8 JSON.
    # - Use indent=2.
    # - Do not suppress OSError exceptions.
    # raise NotImplementedError("Implement write_json_report")
    output_path = Path(output_path)

    # Create the parent directory if it does not exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the report and write it as JSON.
    report = {
        "files": reports,
        "total": total,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="training-log-analyzer",
        description="Count lines, errors, and warnings in training logs.",
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="One or more UTF-8 training log files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for a JSON report.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level. Defaults to INFO.",
    )
    return parser


def main(argv=None):
    """Analyze one or more log files and optionally save a JSON report."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    reports = []
    had_file_error = False

    for path in args.log_files:
        try:
            counts = analyze_file(path)
        except OSError as exc:
            LOGGER.error("Could not read %s: %s", path, exc)
            had_file_error = True
            continue

        report = {"file": str(path), **counts}
        reports.append(report)

        print(f"File: {path}")
        print(f"  total_lines: {counts['total_lines']}")
        print(f"  error_count: {counts['error_count']}")
        print(f"  warning_count: {counts['warning_count']}")

    if not reports:
        LOGGER.error("No input log files could be analyzed.")
        return 1

    # TODO 5:
    # - Compute aggregate counters with aggregate_reports().
    # - Print a TOTAL section with total_lines, error_count, warning_count.
    # - If args.output is not None, call write_json_report().
    # - Log a useful INFO message after a successful report.
    # - If writing the report raises OSError, log it and return 1.
    # - Return 1 if had_file_error is True; otherwise return 0.
    # raise NotImplementedError("Complete main")
    total = aggregate_reports(reports=reports)

    print("********** TOTAL section **********")
    print(f"  total_lines: {total['total_lines']}")
    print(f"  error_count: {total['error_count']}")
    print(f"  warning_count: {total['warning_count']}")
    print("***********************************")

    # Write the JSON report if an output path was provided.
    if args.output is not None:
        try:
            write_json_report(
                reports=reports,
                total=total,
                output_path=args.output,
            )
        except OSError as exc:
            LOGGER.error(
                "Could not write JSON report to %s: %s",
                args.output,
                exc,
            )
            return 1

        LOGGER.info("JSON report written to %s", args.output)

    # Log the final analysis summary.
    LOGGER.info(
        "Analyzed %d file(s): %d lines, %d errors, %d warnings",
        len(reports),
        total['total_lines'],
        total['error_count'],
        total['warning_count'],
    )

    # Return 1 if any input file failed; otherwise return 0.
    return 1 if had_file_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
