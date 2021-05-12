import unittest
import numpy as np
from onnxruntime_extensions.eager_op import EagerOp, StringToVector


def _run_string_to_vector(input, output, map, unk):
    str2vector = EagerOp.from_customop(StringToVector, map=map, unk=unk)
    result = str2vector(input)
    np.testing.assert_array_equal(result, output)


class TestStringToVector(unittest.TestCase):

    def test_string_to_vector1(self):
        _run_string_to_vector(input=np.array(["a", "b", "c", "unknown_word"]),
                              output=np.array([[0], [2], [3], [-1]], dtype=np.int64),
                              map={"a": [0], "b": [2], "c": [3]},
                              unk=[-1])

    def test_string_to_vector2(self):
        _run_string_to_vector(input=np.array(["a", "b", "c", "unknown_word"]),
                              output=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [-1, -1, -1]], dtype=np.int64),
                              map={"a": [0, 1, 2], "b": [1, 2, 3], "c": [2, 3, 4]},
                              unk=[-1, -1, -1])

    def test_string_to_vector3(self):
        _run_string_to_vector(input=np.array(["a", "b", "c", "unknown_word", "你好", "下午", "测试"]),
                              output=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [-1, -1, -1], [6, 6, 6], [7, 8, 9], [-1, -1, -1]], dtype=np.int64),
                              map={"a": [0, 1, 2], "b": [1, 2, 3], "c": [2, 3, 4], "你好": [6, 6, 6], "下午": [7, 8, 9]},
                              unk=[-1, -1, -1])


if __name__ == "__main__":
    unittest.main()
