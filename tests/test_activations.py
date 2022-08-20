from unittest import TestCase
import numpy as np

from learna.activations import (
    binary_step,
    hyperbolic_tangent,
    leaky_rectified_linear_unit,
    linear,
    rectified_linear_unit,
    sigmoid,
    softmax,
)


class TestActvations(TestCase):
    def test_binary_step(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = binary_step(x)
        expected = np.array([0, 0, 0, 1, 1, 1, 1, np.nan])
        np.testing.assert_array_equal(expected, res)

    def test_hyperbolic_tangent(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = hyperbolic_tangent(x)
        expected = np.tanh(x)
        np.testing.assert_array_equal(expected, res)

    def test_leaky_rectified_linear_unit(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = leaky_rectified_linear_unit(x)
        expected = np.where(x > 0, x, x * 0.01)
        np.testing.assert_array_equal(expected, res)

    def test_linear(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = linear(x)
        expected = x
        np.testing.assert_array_equal(expected, res)

    def test_rectified_linear_unit(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = rectified_linear_unit(x)
        expected = np.where(x >= 0, x, 0)
        expected[np.isnan(x)] = np.nan
        np.testing.assert_array_equal(expected, res)

    def test_sigmoid(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_equal(expected, res)

    def test_softmax(self):
        x = np.array([-np.inf, -2, -1.2, 0, 1.2, 2, np.inf, np.nan])
        res = softmax(x)
        expected = np.exp(x) / np.sum(np.exp(x), axis=0)
        np.testing.assert_array_equal(expected, res)
