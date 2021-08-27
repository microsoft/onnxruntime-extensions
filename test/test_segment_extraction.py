from pathlib import Path
import unittest
import numpy as np
from onnxruntime_extensions import PyOrtFunction, SegmentExtraction


def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))


def _run_segment_extraction(input, expect_position, expect_value):
    t2stc = PyOrtFunction.from_customop(SegmentExtraction)
    position, value = t2stc(input)
    np.testing.assert_array_equal(position, expect_position)
    np.testing.assert_array_equal(value, expect_value)


class TestBlingFireSentenceBreaker(unittest.TestCase):

    def test_text_to_case1(self):
        inputs = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int64)
        position = [[2, 3], [4, 6], [7, 10]]
        value = [1, 2, 3]
        _run_segment_extraction(inputs, position, value)

        inputs = np.array([1, 1, 0, 0, 2, 2, 2, 3, 3, 3, 0, 5], dtype=np.int64)
        position = [[0, 1], [4, 6], [7, 9], [11, 11]]
        value = [1, 2, 3, 5]
        _run_segment_extraction(inputs, position, value)

        inputs = np.array([1, 2, 4, 5], dtype=np.int64)
        position = [[0, 0], [1, 1], [2, 2], [3, 3]]
        value = [1, 2, 4, 5]
        _run_segment_extraction(inputs, position, value)


if __name__ == "__main__":
    unittest.main()
