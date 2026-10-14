"""
Python Tutorial - 6: Operating system, CLI, regex, math, and statistics.

The default run uses a temporary directory and requires no arguments.
To inspect the first lines of real files, use:
    python week01_python/6_os_cmd_math.py -l 5 README.md

The stdin example is available as a function but is not called by default.
"""

import argparse
import glob
import math
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path


def operating_system_example():
    """Explore OS information and work with files safely."""
    print("Current directory:", os.getcwd())
    print("Operating system:", os.name)
    print("Example os attributes:", [name for name in dir(os) if name.startswith("get")][:5])

    # Use a temporary directory instead of a hard-coded server path.
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        original_directory = Path.cwd()

        try:
            os.chdir(workspace)
            print("Temporary working directory:", os.getcwd())

            # Prefer pathlib for creating directories over os.system("mkdir ...").
            Path("today").mkdir()

            Path("data.db").write_text("sample data", encoding="utf-8")
            copied = shutil.copyfile("data.db", "archive.db")
            moved = shutil.move(copied, "today/archive.db")
            print("Copied and moved file:", moved)
            print("Matching Python files:", glob.glob("*.py"))
            print("Archive exists:", Path(moved).is_file())
            assert Path(moved).read_text(encoding="utf-8") == "sample data"
        finally:
            os.chdir(original_directory)


def show_top_lines(filenames, number_of_lines=10):
    """Print up to number_of_lines lines from each UTF-8 file."""
    if number_of_lines < 0:
        raise ValueError("number_of_lines must be non-negative")

    for filename in filenames:
        print(f"=== {filename} ===")
        with open(filename, encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index >= number_of_lines:
                    break
                print(line, end="")


def parse_command_line(argv):
    """Parse a file-list command line for the top-lines example."""
    parser = argparse.ArgumentParser(
        prog="top",
        description="Show the top lines from each file.",
    )
    parser.add_argument("filenames", nargs="+")
    parser.add_argument("-l", "--lines", type=int, default=10)
    args = parser.parse_args(argv)
    if args.lines < 0:
        parser.error("--lines must be non-negative")
    return args


def echo_stdin(stream=None):
    """Echo input lines, writing empty-line warnings to stderr.

    Reading stdin can block until input is available, so this function
    is intentionally not invoked by the default demonstration.
    """
    if stream is None:
        stream = sys.stdin

    for line in stream:
        line = line.rstrip("\n")
        if line:
            print(line, file=sys.stdout)
        else:
            print("Empty line", file=sys.stderr)


def text_processing_example():
    """Compare regular expressions and simple string replacement."""
    text = "which foot or hand fell fastest"
    print("Regex matches:", re.findall(r"\bf[a-z]*", text))
    print(
        "Remove repeated word:",
        re.sub(r"(\b[a-z]+) \1", r"\1", "cat in the the hat"),
    )
    print("Simple replacement:", "tea for too".replace("too", "two"))


def math_random_statistics_example():
    """Explore numeric utilities using reproducible random examples."""
    print("cos(pi / 4):", math.cos(math.pi / 4))
    print("log2(1024):", math.log2(1024))

    rng = random.Random(7)
    print("Choice:", rng.choice(["apple", "pear", "banana"]))
    print("Unique sample:", rng.sample(range(100), 10))
    print("Random float:", rng.random())
    print("Random integer:", rng.randrange(6))

    data = [2.75, 1.75, 1.25, 0.25, 0.5, 1.25, 3.5]
    print("Mean:", statistics.mean(data))
    print("Median:", statistics.median(data))
    print("Sample variance:", statistics.variance(data))


def main(argv=None):
    """Run examples, or show top lines when filenames are provided."""
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        args = parse_command_line(argv)
        show_top_lines(args.filenames, args.lines)
        return

    operating_system_example()
    text_processing_example()
    math_random_statistics_example()
    print("All module 6 examples completed!")


if __name__ == "__main__":
    main()
