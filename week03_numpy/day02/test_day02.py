"""Unit tests for Week 03 Day 02 NumPy indexing exercises."""

import unittest

import numpy as np

from week03_numpy.day02.day02 import (
    replace_above_threshold,
    select_by_indices,
    select_by_mask,
    select_matrix_coordinates,
)


class TestSelectByIndices(unittest.TestCase):
    def test_selects_elements_and_returns_independent_result(self):
        array = np.array([10, 20, 30, 40, 50])
        original = array.copy()

        result = select_by_indices(array, [0, 2, 4])

        np.testing.assert_array_equal(result, [10, 30, 50])
        self.assertFalse(np.shares_memory(result, array))
        np.testing.assert_array_equal(array, original)

        result[0] = -1
        self.assertEqual(array[0], 10)

    def test_rejects_non_1d_array(self):
        with self.assertRaises(ValueError):
            select_by_indices(np.arange(6).reshape(2, 3), [0, 1])

    def test_rejects_invalid_indices_shape_or_dtype(self):
        array = np.arange(5)

        with self.assertRaises(ValueError):
            select_by_indices(array, [[0, 1]])

        with self.assertRaises(ValueError):
            select_by_indices(array, [0.0, 1.0])


class TestSelectByMask(unittest.TestCase):
    def test_selects_true_entries_and_returns_independent_result(self):
        array = np.array([-2, 0, 3, 5])
        mask = np.array([False, False, True, True])
        original = array.copy()

        result = select_by_mask(array, mask)

        np.testing.assert_array_equal(result, [3, 5])
        self.assertFalse(np.shares_memory(result, array))
        np.testing.assert_array_equal(array, original)

        result[0] = -1
        self.assertEqual(array[2], 3)

    def test_rejects_non_boolean_mask(self):
        with self.assertRaises(ValueError):
            select_by_mask(np.arange(3), [0, 1, 0])

    def test_rejects_mask_with_wrong_shape(self):
        with self.assertRaises(ValueError):
            select_by_mask(np.arange(3), [True, False])

    def test_rejects_non_1d_array(self):
        with self.assertRaises(ValueError):
            select_by_mask(np.arange(6).reshape(2, 3), np.ones((2, 3), dtype=bool))


class TestReplaceAboveThreshold(unittest.TestCase):
    def test_replaces_only_values_strictly_above_threshold(self):
        array = np.array([1, 3, 4, 5])
        original = array.copy()

        result = replace_above_threshold(array, threshold=3, replacement=0)

        np.testing.assert_array_equal(result, [1, 3, 0, 0])
        np.testing.assert_array_equal(array, original)
        self.assertFalse(np.shares_memory(result, array))

    def test_returns_new_array_even_when_nothing_matches(self):
        array = np.array([1, 2, 3])

        result = replace_above_threshold(array, threshold=10, replacement=0)

        np.testing.assert_array_equal(result, array)
        self.assertFalse(np.shares_memory(result, array))

    def test_rejects_non_1d_array(self):
        with self.assertRaises(ValueError):
            replace_above_threshold(np.arange(6).reshape(2, 3), 3, 0)


class TestSelectMatrixCoordinates(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
        ])

    def test_selects_paired_coordinates(self):
        original = self.matrix.copy()

        result = select_matrix_coordinates(self.matrix, [0, 2], [1, 0])

        np.testing.assert_array_equal(result, [11, 30])
        self.assertEqual(result.shape, (2,))
        self.assertFalse(np.shares_memory(result, self.matrix))
        np.testing.assert_array_equal(self.matrix, original)

    def test_rejects_non_2d_matrix(self):
        with self.assertRaises(ValueError):
            select_matrix_coordinates(np.arange(4), [0], [1])

    def test_rejects_coordinate_length_mismatch(self):
        with self.assertRaises(ValueError):
            select_matrix_coordinates(self.matrix, [0, 1], [1])

    def test_rejects_non_1d_coordinates(self):
        with self.assertRaises(ValueError):
            select_matrix_coordinates(self.matrix, [[0, 1]], [1, 2])

    def test_rejects_non_integer_coordinates(self):
        with self.assertRaises(ValueError):
            select_matrix_coordinates(self.matrix, [0.0, 2.0], [1, 0])


if __name__ == "__main__":
    unittest.main()
