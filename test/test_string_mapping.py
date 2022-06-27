import unittest
import numpy as np
from onnxruntime_extensions import util
from onnxruntime_extensions import PyOrtFunction, StringMapping


def _run_string_mapping(input, output, map):
    v2str = PyOrtFunction.from_customop(StringMapping, map=map)
    result = v2str(input)
    np.testing.assert_array_equal(result, output)


class TestStringMapping(unittest.TestCase):

    def test_string_mapping_case1(self):
        _run_string_mapping(input=np.array(["a", "b", "c", "d"]),
                            output=np.array(["e", "f", "c", "unknown_word"]),
                            map={"a": "e", "b": "f", "d": "unknown_word"})

    def test_string_mapping_case2(self):
        _run_string_mapping(input=np.array(["a", "b", "c", "excel spreadsheet"]),
                            output=np.array(["a", "b", "c", "excel"]),
                            map=util.read_file(util.get_test_data_file("data", "string_mapping.txt")))

    def test_string_mapping_case3(self):
        _run_string_mapping(
            input=np.array(
                ["a", "b", "c", "excel spreadsheet", "image", "imag", "powerpoint presentations",
                 "powerpointpresentation"]),
            output=np.array(
                ["a", "b", "c", "excel", "image", "imag", "ppt", "powerpointpresentation"]),
            map=util.read_file(util.get_test_data_file("data", "string_mapping.txt")))


if __name__ == "__main__":
    unittest.main()
