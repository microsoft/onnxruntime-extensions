# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import os
import onnx
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model
import sys

# add tools dir where gen_customop_template.py script is to sys path
script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, ".."))
tools_dir = os.path.join(ort_ext_root, "tools")
test_data_dir = os.path.join(ort_ext_root, "test", "data")
sys.path.append(tools_dir)

import gen_customop_template    # noqa: E402


# create generic custom op models with some basic math ops for testing purposes
def _create_test_model_1():
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('PyReverseMatrix',
                                  ['identity1'], ['reversed'],
                                  domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [None, 2])
    output0 = helper.make_tensor_value_info(
        'reversed', onnx_proto.TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = make_onnx_model(graph)
    return model


def _create_test_model_2(prefix=""):
    nodes = [
        helper.make_node("Identity", ["data"], ["id1"]),
        helper.make_node("Identity", ["segment_ids"], ["id2"]),
        helper.make_node("%sSegmentSum" % prefix, ["id1", "id2"], ["z"], domain="ai.onnx.contrib"),
    ]

    input0 = helper.make_tensor_value_info("data", onnx_proto.TensorProto.FLOAT, [])
    input1 = helper.make_tensor_value_info(
        "segment_ids", onnx_proto.TensorProto.INT64, []
    )
    output0 = helper.make_tensor_value_info("z", onnx_proto.TensorProto.FLOAT, [])

    graph = helper.make_graph(nodes, "test0", [input0, input1], [output0])
    model = make_onnx_model(graph)
    return model


class TestCustomOpTemplate(unittest.TestCase):

    @classmethod
    def tearDownClass(cls) -> None:
        # remove generated files
        template_output_path = os.path.join(test_data_dir, "generated")
        if os.path.exists(template_output_path):
            for file in os.listdir(template_output_path):
                os.remove(os.path.join(template_output_path, file))
            os.rmdir(template_output_path)
        return super().tearDownClass()

    # check input and output type count of models extracted by template generator
    def check_io_count(self, model_name, output_path, expected_input_count, expected_output_count):
        model_path = os.path.join(test_data_dir, "generated", model_name)
        input_count, output_count = gen_customop_template.main([model_path, output_path])
        self.assertEqual(input_count, expected_input_count)
        self.assertEqual(output_count, expected_output_count)

    def test_template(self):
        template_output_path = os.path.join(test_data_dir, "generated")
        os.mkdir(template_output_path)

        onnx.save(_create_test_model_1(), os.path.join(template_output_path, "test_model_1.onnx"))
        test1_template_output_path = os.path.join(template_output_path, "custom_op_template_test1.hpp")
        self.check_io_count(model_name="test_model_1.onnx",
                            output_path=test1_template_output_path,
                            expected_input_count=1, expected_output_count=1)

        onnx.save(_create_test_model_2(), os.path.join(template_output_path, "test_model_2.onnx"))
        test2_template_output_path = os.path.join(template_output_path, "custom_op_template_test2.hpp")
        self.check_io_count(model_name="test_model_2.onnx",
                            output_path=test2_template_output_path,
                            expected_input_count=2, expected_output_count=1)


if __name__ == "__main__":
    unittest.main()
