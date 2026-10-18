import argparse
import logging

from week02_system.day05.metrics import summarize_step_times


LOGGER = logging.getLogger(__name__)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="step-summary",
        description="Summarize training step durations.",
    )

    # TODO 1: Add the required --times argument.
    # It should accept one or more float values.
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        required=True,
        help="Training step durations in seconds.",
    )

    # TODO 2: Add the --log_level argument.
    # Choices: DEBUG, INFO, WARNING, ERROR.
    # Default: INFO.
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level. Defaults to INFO.",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # TODO 3: Configure logging based on args.log_level.
    logging.basicConfig(level=args.log_level)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )
    LOGGER.debug("Received step durations: %s", args.times)

    try:
        summary = summarize_step_times(args.times)
    except ValueError as exc:
        # TODO 4: Log the error and return exit code 2.
        LOGGER.error("Could not summarize step times: %s", exc)
        return 2

    # TODO 5: Log a successful summary.
    # Print count, mean duration, and maximum duration.
    # Return exit code 0.
    LOGGER.info(
        "Processed %d training steps",
        summary["count"],
    )
    LOGGER.debug("Summary details: %s", summary)

    print(f"step_count: {summary['count']}")
    print(
        "mean_step_time_seconds: "
        f"{summary['mean_seconds']:.3f}"
    )
    print(
        "max_step_time_seconds: "
        f"{summary['max_seconds']:.3f}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())