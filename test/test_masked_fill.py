import unittest
import numpy as np
from onnxruntime_extensions import PyOrtFunction, MaskedFill


def read_file(path):
    with open(path) as file_content:
        return file_content.read()


def run_string_remove(value, mask, output):
    v2str = PyOrtFunction.from_customop(MaskedFill)
    result = v2str(value, mask)
    np.testing.assert_array_equal(result, output)


class TestMaskedFill(unittest.TestCase):

    def test_string_remove_case1(self):
        run_string_remove(value=np.array(["a", "b", "c", "d"]), mask=np.array([True, True, False, True], dtype=bool),
                          output=np.array(["a", "b", "d"]))
        run_string_remove(value=np.array(["a", "b", "c", "d"]), mask=np.array([True, True, True, True], dtype=bool),
                          output=np.array(["a", "b", "c", "d"]))
        run_string_remove(value=np.array(["a", "b", "c", "d"]), mask=np.array([False, False, False, False], dtype=bool),
                          output=np.array([]))

    def test_string_remove_case2(self):
        run_string_remove(value=np.array(["a"]), mask=np.array([False], dtype=bool),
                          output=np.array([]))
        run_string_remove(value=np.array(["a"]), mask=np.array([True], dtype=bool),
                          output=np.array(["a"]))

    def test_string_remove_case3(self):
        run_string_remove(value=np.array(["♠", "♣", "♥", "♦"]), mask=np.array([False, True, True, False], dtype=bool),
                          output=np.array(["♣", "♥"]))
        run_string_remove(
            value=np.array(["As", "fast", "as", "thou", "shalt", "wane", "", "so", "fast", "thou", "grow’st"]),
            mask=np.array([True, True, True, True, True, True, False, True, True, True, True], dtype=bool),
            output=np.array(["As", "fast", "as", "thou", "shalt", "wane", "so", "fast", "thou", "grow’st"]))
