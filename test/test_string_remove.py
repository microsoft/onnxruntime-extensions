from pathlib import Path
import unittest
import numpy as np
from onnxruntime_extensions import PyOrtFunction, StringRemove


def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))


def read_file(path):
    with open(path) as file_content:
        return file_content.read()


def _run_string_remove(strings, conditions, output):
    v2str = PyOrtFunction.from_customop(StringRemove, )
    result = v2str(strings, conditions)
    np.testing.assert_array_equal(result, output)


class TestStringRemove(unittest.TestCase):

    def test_string_remove_case1(self):
        _run_string_remove(strings=np.array(["a", "b", "c", "d"]), conditions=np.array([1, 1, 0, 1], dtype=np.int64),
                           output=np.array(["a", "b", "d"]))
        _run_string_remove(strings=np.array(["a", "b", "c", "d"]), conditions=np.array([1, 1, 1, 1], dtype=np.int64),
                           output=np.array(["a", "b", "c", "d"]))
        _run_string_remove(strings=np.array(["a", "b", "c", "d"]), conditions=np.array([0, 0, 0, 0], dtype=np.int64),
                           output=np.array([]))

    def test_string_remove_case2(self):
        _run_string_remove(strings=np.array(["a"]), conditions=np.array([0], dtype=np.int64),
                           output=np.array([]))
        _run_string_remove(strings=np.array(["a"]), conditions=np.array([1], dtype=np.int64),
                           output=np.array(["a"]))

    def test_string_remove_case3(self):
        _run_string_remove(strings=np.array(["♠", "♣", "♥", "♦"]), conditions=np.array([0, 1, 1, 0], dtype=np.int64),
                           output=np.array(["♣", "♥"]))
        _run_string_remove(
            strings=np.array(["As", "fast", "as", "thou", "shalt", "wane", "", "so", "fast", "thou", "grow’st"]),
            conditions=np.array([2, 4, 2, 4, 5, 4, 0, 2, 4, 4, 7], dtype=np.int64),
            output=np.array(["As", "fast", "as", "thou", "shalt", "wane", "so", "fast", "thou", "grow’st"]))
