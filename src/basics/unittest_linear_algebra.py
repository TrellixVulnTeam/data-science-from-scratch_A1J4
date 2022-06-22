import unittest

import linear_algebra


class TestLinearAlgebra(unittest.TestCase):

    def test_add(self):
        result = linear_algebra.add([1, 2, 3], [4, 5, 6])
        expected_result = [5, 7, 9]
        self.assertEqual(result, expected_result)


    def test_subtract(self):
        result = linear_algebra.subtract([5, 7, 9], [4, 5, 6])
        expected_result = [1, 2, 3]
        self.assertEqual(result, expected_result)


    def test_vector_sum(self):
        result = linear_algebra.vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_result = [16, 20]
        self.assertEqual(result, expected_result)


    def test_scalar_multiply(self):
        result = linear_algebra.scalar_multiply(2, [1, 2, 3])
        expected_result = [2, 4, 6]
        self.assertEqual(result, expected_result)


    def test_vector_mean(self):
        result = linear_algebra.vector_mean([[1, 2], [3, 4], [5, 6]])
        expected_result = [3, 4]
        self.assertEqual(result, expected_result)
 

    def test_dot(self):
        result = linear_algebra.dot([1, 2, 3], [4, 5, 6])  # 1 * 4 + 2 * 5 + 3 * 6
        expected_result = 32
        self.assertEqual(result, expected_result)
 

    def test_sum_of_squares(self):
        result = linear_algebra.sum_of_squares([1, 2, 3])  # 1 * 1 + 2 * 2 + 3 * 3
        expected_result = 14
        self.assertEqual(result, expected_result)
 

    def test_magnitude(self):
        result = linear_algebra.magnitude([3, 4])
        expected_result = 5
        self.assertEqual(result, expected_result)
 

    def test_shape(self):
        result = linear_algebra.shape([[1, 2, 3], [4, 5, 6]])
        expected_result = (2, 3)
        self.assertEqual(result, expected_result)
 

    def test_identity_matrix(self):
        result = linear_algebra.identity_matrix(5)
        expected_result = [[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]]
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()