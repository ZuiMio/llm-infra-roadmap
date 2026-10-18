"""
Python Tutorial - 7: Network, dates, compression, timing, and tests.

Network access and email sending are opt-in functions. Running this file
does not contact websites, send email, or require a local SMTP server.
"""

import datetime as dt
import doctest
import smtplib
import unittest
import zlib
from timeit import Timer
from urllib.request import urlopen


def fetch_python_documentation():
    """Fetch documentation and print matching lines (requires network).

    Call explicitly to perform a network request. Not run by main().
    """
    with urlopen("https://docs.python.org/3/", timeout=10) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace")
            if "updated" in line.lower():
                print(line.rstrip())


def send_example_email():
    """Send a demonstration email through a local SMTP server.

    Call explicitly only if an SMTP server has been configured locally.
    Not run by main() because it performs an external side effect.
    """
    message = (
        "To: jcaesar@example.org\n"
        "From: soothsayer@example.org\n"
        "\n"
        "Beware the Ides of March.\n"
    )
    with smtplib.SMTP("localhost", timeout=10) as server:
        server.sendmail(
            "soothsayer@example.org",
            "jcaesar@example.org",
            message,
        )


def date_example():
    """Demonstrate date formatting and date arithmetic."""
    today = dt.date.today()
    print("Today:", today)
    print(today.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))

    sample_date = dt.date(2004, 1, 4)
    elapsed = today - sample_date
    print("Days since the sample date:", elapsed.days)


def compression_example():
    """Compress bytes, restore them, and calculate a CRC32 checksum."""
    data = b"witch which has which witches wrist watch"
    compressed = zlib.compress(data)
    restored = zlib.decompress(compressed)

    print("Original bytes:", len(data))
    print("Compressed bytes:", len(compressed))
    print("CRC32:", zlib.crc32(data))
    assert restored == data


def timing_example():
    """Compare two ways of swapping values using small benchmarks."""
    number = 100_000
    with_temp = Timer("t = a; a = b; b = t", "a = 1; b = 2").timeit(number)
    unpacking = Timer("a, b = b, a", "a = 1; b = 2").timeit(number)
    print("Swap using temporary variable:", with_temp)
    print("Swap using tuple unpacking:", unpacking)
    # Timing can vary from run to run; do not assert which is faster.


def average(values):
    """Return the arithmetic mean of a nonempty sequence.

    >>> average([20, 30, 70])
    40.0
    >>> round(average([1, 5, 7]), 1)
    4.3
    """
    return sum(values) / len(values)


class TestStatisticalFunctions(unittest.TestCase):
    """Check expected values and invalid usage."""

    def test_average(self):
        self.assertEqual(average([20, 30, 70]), 40.0)
        self.assertAlmostEqual(average([1, 5, 7]), 13 / 3)

    def test_invalid_usage(self):
        with self.assertRaises(ZeroDivisionError):
            average([])
        with self.assertRaises(TypeError):
            average(20, 30, 70)
        with self.assertRaises(TypeError):
            average(None)


def run_tests():
    """Run the doctest and unittest examples without exiting the script."""
    doctest_result = doctest.testmod()
    assert doctest_result.failed == 0

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestStatisticalFunctions
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Unit tests did not pass")


def main():
    """Run offline demonstrations and tests."""
    print("=== Dates ===")
    date_example()

    print("\n=== Compression ===")
    compression_example()

    print("\n=== Timing ===")
    timing_example()

    print("\n=== Testing ===")
    run_tests()

    print("All module 7 offline examples completed!")
    print("Network and email examples are available as opt-in functions.")


if __name__ == "__main__":
    main()
