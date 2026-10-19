"""Unit tests for the training log analyzer."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from week02_system.day07.log_analyzer import (
    aggregate_reports,
    analyze_file,
    analyze_lines,
    main,
)


class TestAnalyzeLines(unittest.TestCase):
    def test_counts_markers_case_insensitively(self):
        lines = [
            "INFO started\n",
            "error: out of memory\n",
            "WARNING: slow batch\n",
            "ERROR and WARNING on the same line\n",
            "regular message\n",
        ]

        result = analyze_lines(lines)

        self.assertEqual(
            result,
            {
                "total_lines": 5,
                "error_count": 2,
                "warning_count": 2,
            },
        )

    def test_empty_input(self):
        self.assertEqual(
            analyze_lines([]),
            {
                "total_lines": 0,
                "error_count": 0,
                "warning_count": 0,
            },
        )


class TestAnalyzeFile(unittest.TestCase):
    def test_reads_utf8_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.log"
            path.write_text(
                "INFO started\nERROR failed\nWARNING slow\n",
                encoding="utf-8",
            )

            result = analyze_file(path)

        self.assertEqual(
            result,
            {
                "total_lines": 3,
                "error_count": 1,
                "warning_count": 1,
            },
        )


class TestAggregateReports(unittest.TestCase):
    def test_sums_multiple_reports(self):
        reports = [
            {
                "file": "one.log",
                "total_lines": 2,
                "error_count": 1,
                "warning_count": 0,
            },
            {
                "file": "two.log",
                "total_lines": 3,
                "error_count": 0,
                "warning_count": 2,
            },
        ]

        self.assertEqual(
            aggregate_reports(reports),
            {
                "total_lines": 5,
                "error_count": 1,
                "warning_count": 2,
            },
        )

    def test_empty_reports(self):
        self.assertEqual(
            aggregate_reports([]),
            {
                "total_lines": 0,
                "error_count": 0,
                "warning_count": 0,
            },
        )


class TestCommandLine(unittest.TestCase):
    def test_multiple_files_and_json_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.log"
            second = root / "second.log"
            output_path = root / "reports" / "summary.json"

            first.write_text("INFO start\nERROR oom\n", encoding="utf-8")
            second.write_text(
                "WARNING slow\nINFO complete\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        str(first),
                        str(second),
                        "--output",
                        str(output_path),
                        "--log-level",
                        "ERROR",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("TOTAL", stdout.getvalue())
            self.assertEqual(output_path.exists(), True)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["files"]), 2)
            self.assertEqual(
                payload["total"],
                {
                    "total_lines": 4,
                    "error_count": 1,
                    "warning_count": 1,
                },
            )

    def test_missing_file_returns_error_but_does_not_crash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.log"
            missing_path = Path(temp_dir) / "missing.log"
            valid_path.write_text("ERROR test\n", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                with self.assertLogs(
                    "week02_system.day07.log_analyzer",
                    level="ERROR",
                ) as captured:
                    exit_code = main(
                        [str(valid_path), str(missing_path)]
                    )

            self.assertEqual(exit_code, 1)
            self.assertIn("File:", stdout.getvalue())
            self.assertIn("Could not read", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
