# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import os
import sys

# add tools dir where gen_customop_template.py script is to sys path
script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, ".."))
tools_dir = os.path.join(ort_ext_root, "tools")
test_data_dir = os.path.join(ort_ext_root, "test", "data")
sys.path.append(tools_dir)

import gen_customop_template

class TestCustomOpTemplate(unittest.TestCase):

    # check input and output type count of models extracted by template generator
    def check_io_count(self, model_name, output_path, expected_input_count, expected_output_count):
        model_path = os.path.join(test_data_dir, model_name)
        input_count, output_count = gen_customop_template.main([model_path, output_path])
        self.assertEqual(input_count, expected_input_count)
        self.assertEqual(output_count, expected_output_count)

    def test_template(self):
        gpt2tok_template_output_path = os.path.join(tools_dir, "custom_op_template_gpt2tok.hpp")
        self.check_io_count(model_name = "custom_op_gpt2tok.onnx", output_path = gpt2tok_template_output_path, expected_input_count = 1, expected_output_count = 2)
        
        cliptok_template_output_path = os.path.join(tools_dir, "custom_op_template_cliptok.hpp")
        self.check_io_count(model_name = "custom_op_cliptok.onnx", output_path = cliptok_template_output_path, expected_input_count = 1, expected_output_count = 2)

if __name__ == "__main__":
    unittest.main()