"""Unit tests for Week 03 Day 01 NumPy layout exercises."""

# python -m unittest week03_numpy.day01.test_day01
import unittest

import numpy as np

from week03_numpy.day01.day01 import (
    describe_array,
    slice_view_and_copy,
    transpose_layout,
)


class TestDescribeArray(unittest.TestCase):
    def test_describes_c_contiguous_array(self):
        array = np.arange(12, dtype=np.float64).reshape(3, 4)

        result = describe_array(array)

        self.assertEqual(result["shape"], (3, 4))
        self.assertEqual(result["ndim"], 2)
        self.assertEqual(result["dtype"], "float64")
        self.assertEqual(result["strides"], (32, 8))
        self.assertEqual(result["nbytes"], 96)
        self.assertIs(result["c_contiguous"], True)

    def test_transposed_array_has_different_layout(self):
        array = np.arange(12, dtype=np.float64).reshape(3, 4)

        result = describe_array(array.T)

        self.assertEqual(result["shape"], (4, 3))
        self.assertEqual(result["strides"], (8, 32))
        self.assertEqual(result["nbytes"], 96)
        self.assertIs(result["c_contiguous"], False)


class TestSliceViewAndCopy(unittest.TestCase):
    def test_view_shares_memory_but_copy_does_not(self):
        matrix = np.arange(12).reshape(3, 4)
        original = matrix.copy()

        view, copied_slice = slice_view_and_copy(matrix)

        self.assertEqual(view.shape, (3, 2))
        np.testing.assert_array_equal(view, original[:, 1:3])
        np.testing.assert_array_equal(copied_slice, original[:, 1:3])

        self.assertTrue(np.shares_memory(view, matrix))
        self.assertFalse(np.shares_memory(copied_slice, matrix))

        view[0, 0] = -100
        self.assertEqual(matrix[0, 1], -100)
        self.assertEqual(copied_slice[0, 0], original[0, 1])

        copied_slice[1, 1] = -200
        self.assertEqual(matrix[1, 2], original[1, 2])

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            slice_view_and_copy(np.arange(6))

        with self.assertRaises(ValueError):
            slice_view_and_copy(np.ones((2, 2)))


class TestTransposeLayout(unittest.TestCase):
    def test_transpose_is_view_and_copy_is_c_contiguous(self):
        matrix = np.arange(6).reshape(2, 3)
        original = matrix.copy()

        transposed, contiguous_copy = transpose_layout(matrix)

        self.assertEqual(transposed.shape, (3, 2))
        np.testing.assert_array_equal(transposed, original.T)
        np.testing.assert_array_equal(contiguous_copy, original.T)

        self.assertTrue(np.shares_memory(transposed, matrix))
        self.assertFalse(np.shares_memory(contiguous_copy, matrix))
        self.assertFalse(transposed.flags["C_CONTIGUOUS"])
        self.assertTrue(contiguous_copy.flags["C_CONTIGUOUS"])

        transposed[0, 1] = -10
        self.assertEqual(matrix[1, 0], -10)

    def test_rejects_non_matrix_input(self):
        with self.assertRaises(ValueError):
            transpose_layout(np.arange(5))


if __name__ == "__main__":
    unittest.main()
