"""
Python Tutorial - 9: Threading, logging, and weak references.

The thread example creates its input and ZIP archive in a temporary
directory, so it does not depend on files in the project repository.
"""

import gc
import logging
import tempfile
import threading
import weakref
import unittest
import zipfile
from pathlib import Path


class AsyncZip(threading.Thread):
    """Create a ZIP archive in a separate thread."""

    def __init__(self, infile, outfile):
        super().__init__()
        self.infile = infile
        self.outfile = outfile

    def run(self):
        with zipfile.ZipFile(self.outfile, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.infile, arcname=Path(self.infile).name)
        print("Finished background zip of:", self.infile)


def threading_example():
    """Start a worker thread and wait for it to finish."""
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "mydata.txt"
        archive_path = Path(directory) / "myarchive.zip"
        source.write_text("An example text file.\n", encoding="utf-8")

        background = AsyncZip(source, archive_path)
        background.start()
        print("The main program continues in the foreground.")

        background.join()
        print("The main program waited until the worker finished.")

        with zipfile.ZipFile(archive_path, "r") as archive:
            assert archive.read(source.name) == source.read_bytes()


def logging_example():
    """Demonstrate the standard five common logging levels."""
    logger = logging.getLogger("python_tutorial.module9")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        logger.debug("Debugging information")
        logger.info("Informational message")
        logger.warning("Warning: config file %s not found", "server.conf")
        logger.error("Error occurred")
        logger.critical("Critical error -- shutting down")
    finally:
        logger.removeHandler(handler)
        handler.close()


class A:
    """A simple object used to demonstrate reference lifetimes."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


def weak_reference_example():
    """Compare a strong dictionary reference with a weak reference."""
    normal = {}
    item = A(10)
    normal["primary"] = item
    del item
    gc.collect()

    print("Normal dict keeps its value:", normal["primary"])
    assert normal["primary"].value == 10

    weak = weakref.WeakValueDictionary()
    item = A(20)
    weak["primary"] = item
    print("Weak dict before deletion:", weak.get("primary"))
    assert weak.get("primary") is item

    # Avoid keeping another strong reference such as:
    # saved = weak["primary"].
    # That would prevent the object from being collected.
    del item
    gc.collect()
    print("Weak dict after deletion:", weak.get("primary"))
    assert weak.get("primary") is None


def main():
    """Run all examples."""
    print("=== Threading ===")
    threading_example()
    print("\n=== Logging ===")
    logging_example()
    print("\n=== Weak references ===")
    weak_reference_example()
    print("\\n=== Tests ===")
    run_tests()
    print("All module 9 examples and tests passed!")


class TestReferenceBehavior(unittest.TestCase):
    """Verify strong and weak references without depending on log output."""

    def test_normal_dict_keeps_object_alive(self):
        normal = {}
        item = A(10)
        normal["primary"] = item
        del item
        gc.collect()
        self.assertEqual(normal["primary"].value, 10)

    def test_weak_value_dictionary_drops_unreferenced_object(self):
        weak = weakref.WeakValueDictionary()
        item = A(20)
        weak["primary"] = item
        self.assertIs(weak.get("primary"), item)
        del item
        gc.collect()
        self.assertIsNone(weak.get("primary"))

    def test_repr(self):
        self.assertEqual(repr(A(42)), "42")


def run_tests():
    """Run deterministic weak-reference tests."""
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestReferenceBehavior)
    )
    if not result.wasSuccessful():
        raise AssertionError("Module 9 tests failed")


if __name__ == "__main__":
    main()
