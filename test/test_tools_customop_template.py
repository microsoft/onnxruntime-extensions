# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import os
import sys

# add tools dir where gen_customop_template.py script is to sys path
script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, ".."))
tools_dir = os.path.join(ort_ext_root, "tools")
sys.path.append(tools_dir)

import gen_customop_template

class TestCustomOpTemplate(unittest.TestCase):

    # check input and output type count of models extracted by template generator
    def check_io_count(self, model_path, output_path, expected_input_count, expected_output_count):
        input_count, output_count = gen_customop_template.main([model_path, output_path])
        self.assertEqual(input_count, expected_input_count)
        self.assertEqual(output_count, expected_output_count)

    def test_template(self):
        self.check_io_count(model_path = "test\data\custom_op_gpt2tok.onnx", output_path = "tools\custom_op_template_gpt2tok.hpp", expected_input_count = 1, expected_output_count = 2)
        self.check_io_count(model_path = "test\data\custom_op_cliptok.onnx", output_path = "tools\custom_op_template_cliptok.hpp", expected_input_count = 1, expected_output_count = 2)

if __name__ == "__main__":
    unittest.main()