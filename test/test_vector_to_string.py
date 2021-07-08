import unittest
import numpy as np
from onnxruntime_extensions import PyOrtFunction, VectorToString


def _run_vector_to_string(input, output, map, unk):
    v2str = PyOrtFunction.from_customop(VectorToString, map=map, unk=unk)
    result = v2str(input)
    np.testing.assert_array_equal(result, output)


class TestVectorToString(unittest.TestCase):

    def test_vector_to_case1(self):
        _run_vector_to_string(input=np.array([0, 2, 3, 4], dtype=np.int64),
                              output=np.array(["a", "b", "c", "unknown_word"]),
                              map={"a": [0], "b": [2], "c": [3]},
                              unk="unknown_word")

    def test_vector_to_case2(self):
        _run_vector_to_string(input=np.array([[0, ], [2, ], [3, ], [4, ]], dtype=np.int64),
                              output=np.array(["a", "b", "c", "unknown_word"]),
                              map={"a": [0], "b": [2], "c": [3]},
                              unk="unknown_word")

    def test_vector_to_case3(self):
        _run_vector_to_string(input=np.array([[0, 1], [2, 3], [3, 4], [4, 5]], dtype=np.int64),
                              output=np.array(["a", "b", "c", "unknown_word"]),
                              map={"a": [0, 1], "b": [2, 3], "c": [3, 4]},
                              unk="unknown_word")


if __name__ == "__main__":
    unittest.main()
