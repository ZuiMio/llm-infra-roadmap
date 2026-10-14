"""
Python Tutorial - 8: Text formatting, binary data, and ZIP archives.

Examples create ZIP data inside a temporary directory. No external
files are required. Locale examples fall back gracefully if a
particular locale is unavailable on this computer.
"""

import locale
import os.path
import pprint
import reprlib
import struct
import tempfile
import textwrap
import time
import zipfile
from pathlib import Path
from string import Template


def shortened_representation_example():
    """Use reprlib to abbreviate large object representations."""
    characters = set("supercalifragilisticexpialidocious")
    print("Full representation:", characters)
    print("Short representation:", reprlib.repr(characters))


def pretty_print_example():
    """Pretty-print a nested structure with constrained line width."""
    data = [[[["black", "cyan"], "white", ["green", "red"]],
             [["magenta", "yellow"], "blue"]]]
    pprint.pprint(data, width=30)


def text_wrap_example():
    """Wrap a paragraph into lines of a given width."""
    doc = (
        "The wrap() method is just like fill() except that it returns "
        "a list of strings instead of one big string with newlines "
        "to separate the wrapped lines."
    )
    print(textwrap.fill(doc, width=40))
    print("Wrapped line count:", len(textwrap.wrap(doc, width=40)))


def locale_example():
    """Demonstrate locale-aware numeric formatting where available."""
    original_locale = locale.setlocale(locale.LC_ALL)
    try:
        try:
            selected = locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        except locale.Error:
            # This locale is not installed on every operating system.
            selected = original_locale
            print("en_US.UTF-8 is unavailable; using:", selected)

        print("Active locale:", selected)
        number = 1234567.8
        print("Localized integer:", locale.format_string("%d", number, grouping=True))
        print("Localized float:", locale.format_string("%.2f", number, grouping=True))
        print("Locale-independent formatting:", f"{number:,.2f}")

        # Monetary formatting depends on a configured monetary locale.
        conventions = locale.localeconv()
        print("Currency symbol:", repr(conventions["currency_symbol"]))
    finally:
        locale.setlocale(locale.LC_ALL, original_locale)


def string_template_example():
    """Substitute template fields, including escaped dollar signs."""
    message = Template("$" + "{village}folk send $$10 to $cause.")
    print(message.substitute(village="Nottingham", cause="the ditch fund"))

    missing_value = Template("Return the $item to $owner.")
    values = {"item": "unladen swallow"}
    try:
        print(missing_value.substitute(values))
    except KeyError as error:
        print("Missing template key:", error)
        print("Safe substitution:", missing_value.safe_substitute(values))


class BatchRename(Template):
    """Use percent-prefixed placeholders for file rename previews."""

    delimiter = "%"


def batch_rename_example():
    """Preview renamed files without changing anything on disk."""
    photos = ["img_1074.jpg", "img_1076.jpg", "img_1077.jpg"]
    fmt = "%{n}_%d%f"
    template = BatchRename(fmt)
    date = time.strftime("%d%b%y")

    for index, filename in enumerate(photos):
        _, extension = os.path.splitext(filename)
        new_name = template.substitute(d=date, n=index, f=extension)
        print(f"{filename} --> {new_name}")


def binary_struct_example():
    """Pack and unpack fixed-width binary integers."""
    fields = (0x12345678, 10, 20, 5, 0)
    payload = struct.pack("<IIIHH", *fields)
    unpacked = struct.unpack("<IIIHH", payload)
    print("Packed bytes:", payload)
    print("Unpacked fields:", unpacked)
    assert len(payload) == 16
    assert unpacked == fields


def zip_archive_example():
    """Inspect ZIP entry metadata without requiring an existing ZIP."""
    with tempfile.TemporaryDirectory() as directory:
        archive_path = Path(directory) / "myfile.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("sample.txt", "Hello from a ZIP archive!")
            archive.writestr("notes/info.txt", "Another file.")

        with zipfile.ZipFile(archive_path, "r") as archive:
            for info in archive.infolist():
                print(
                    info.filename,
                    hex(info.CRC),
                    info.compress_size,
                    info.file_size,
                    info.compress_type,
                    info.flag_bits,
                )

            assert archive.read("sample.txt") == b"Hello from a ZIP archive!"


def main():
    """Run all examples."""
    print("=== reprlib ===")
    shortened_representation_example()
    print("\n=== pprint ===")
    pretty_print_example()
    print("\n=== textwrap ===")
    text_wrap_example()
    print("\n=== locale ===")
    locale_example()
    print("\n=== string.Template ===")
    string_template_example()
    print("\n=== BatchRename ===")
    batch_rename_example()
    print("\n=== struct ===")
    binary_struct_example()
    print("\n=== zipfile ===")
    zip_archive_example()
    print("All module 8 examples passed!")


if __name__ == "__main__":
    main()
