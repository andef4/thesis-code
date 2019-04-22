import hdm
import numpy as np


def test_absolute():
    result = hdm.HDMResult(np.array([[1.0, 3.0], [4.0, 5.0]]), baseline=2.5,
                           image_width=3, image_height=3, circle_size=5, offset=5)
    distances = result.distances(hdm.ABSOLUTE)
    assert np.array_equal(distances, np.array([[1.5, 0.5], [1.5, 2.5]]))


def test_better():
    result = hdm.HDMResult(np.array([[1.0, 3.0], [4.0, 5.0]]), baseline=2.5,
                           image_width=3, image_height=3, circle_size=5, offset=5)
    distances = result.distances(hdm.BETTER_ONLY)
    assert np.array_equal(distances, np.array([[1.5, 0.0], [0.0, 0.0]]))


def test_worse():
    result = hdm.HDMResult(np.array([[1.0, 3.0], [4.0, 5.0]]), baseline=2.5,
                           image_width=3, image_height=3, circle_size=5, offset=5)
    distances = result.distances(hdm.WORSE_ONLY)
    assert np.array_equal(distances, np.array([[2.5, 3.0], [4.0, 5.0]]))
