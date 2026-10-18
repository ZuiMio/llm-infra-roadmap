import unittest
from contextlib import redirect_stdout
from contextlib import redirect_stderr
from io import StringIO

from week02_system.day05.metrics import summarize_step_times
from week02_system.day06.app import build_parser, main


class TestMetrics(unittest.TestCase):
    def test_valid_summary(self):
        # TODO: Test count, mean, and maximum.
        result = summarize_step_times([1.0, 2.0, 3.0])

        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["mean_seconds"], 2.0)
        self.assertAlmostEqual(result["max_seconds"], 3.0)

    def test_empty_input(self):
        # TODO: Check that invalid empty input raises ValueError.
        with self.assertRaises(ValueError):
            summarize_step_times([])

    def test_negative_duration(self):
        with self.assertRaises(ValueError):
            summarize_step_times([0.5, -0.1])

    def test_non_finite_duration(self):
        with self.assertRaises(ValueError):
            summarize_step_times([0.5, float("nan")])


class TestCommandLine(unittest.TestCase):
    def test_parse_arguments(self):
        # TODO: Test --times and --log-level.
        args = build_parser().parse_args(
            [
                "--times",
                "0.8",
                "1.2",
                "--log-level",
                "DEBUG",
            ]
        )

        self.assertEqual(args.times, [0.8, 1.2])
        self.assertEqual(args.log_level, "DEBUG")

    def test_missing_times(self):
        stderr = StringIO()

        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as context:
                build_parser().parse_args([])

        self.assertEqual(context.exception.code, 2)

    def test_successful_run(self):
        # TODO: Capture stdout and verify the summary.
        stdout = StringIO()

        with redirect_stdout(stdout):
            with self.assertLogs(
                "week02_system.day06.app",
                level="INFO",
            ) as captured:
                exit_code = main(
                    [
                        "--times",
                        "1.0",
                        "2.0",
                        "3.0",
                        "--log-level",
                        "INFO",
                    ]
                )

        self.assertEqual(exit_code, 0)

        output = stdout.getvalue()
        self.assertIn("step_count: 3", output)
        self.assertIn("mean_step_time_seconds: 2.000", output)
        self.assertIn("max_step_time_seconds: 3.000", output)

        log_output = "\n".join(captured.output)
        self.assertIn("Processed 3 training steps", log_output)

    def test_negative_duration(self):
        # TODO: Verify that the program returns an error code
        # and records an error log.
        stdout = StringIO()

        with redirect_stdout(stdout):
            with self.assertLogs(
                "week02_system.day06.app",
                level="ERROR",
            ) as captured:
                exit_code = main(
                    [
                        "--times",
                        "0.8",
                        "-0.3",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("non-negative", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()