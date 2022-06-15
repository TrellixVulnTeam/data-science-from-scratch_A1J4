import unittest
import operator

import tensor


class TestLinearAlgebra(unittest.TestCase):

    def test_shape_1d(self):
        result = tensor.shape([1, 2, 3])
        expected_result = [3]
        self.assertEqual(result, expected_result)


    def test_shape_2d(self):
        result = tensor.shape([[1, 2], [3, 4], [5, 6]])
        expected_result = [3, 2]
        self.assertEqual(result, expected_result)


    def test_is_1d_true(self):
        result = tensor.is_1d([1, 2, 3])
        self.assertTrue(result)


    def test_is_1d_false(self):
        result = tensor.is_1d([[1, 2], [3, 4]])
        self.assertFalse(result)


    def test_sum_1d(self):
        result = tensor.tensor_sum([1, 2, 3])
        expected_result = 6
        self.assertEqual(result, expected_result)


    def test_sum_2d(self):
        result = tensor.tensor_sum([[1, 2], [3, 4], [5, 6]])
        expected_result = 21
        self.assertEqual(result, expected_result)


    def test_tensor_apply_1d(self):
        result = tensor.tensor_apply(lambda x: x + 1, [1, 2, 3])
        expected_result = [2, 3, 4]
        self.assertEqual(result, expected_result)


    def test_tensor_apply_2d(self):
        result = tensor.tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]])
        expected_result = [[2, 4], [6, 8]]
        self.assertEqual(result, expected_result)


    def test_zeros_like_1d(self):
        result = tensor.zeros_like([1, 2, 3])
        expected_result = [0, 0, 0]
        self.assertEqual(result, expected_result)


    def test_zeros_like_2d(self):
        result = tensor.zeros_like([[1, 2], [3, 4]])
        expected_result = [[0, 0], [0, 0]]
        self.assertEqual(result, expected_result)


    def test_tensor_combine_add(self):
        result = tensor.tensor_combine(operator.add, [1, 2, 3], [4, 5, 6])
        expected_result = [5, 7, 9]
        self.assertEqual(result, expected_result)


    def test_tensor_combine_mul(self):
        result = tensor.tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6])
        expected_result = [4, 10, 18]
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()