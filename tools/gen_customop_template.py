'''
Generates the C++ template code for a new custom operator.

Usage: $ python tools\gen_customop_template.py onnx_model_filepath customop_template_filepath
'''

import argparse
import onnx
import pathlib
import sys

# Load ONNX model
def load_onnx_model(model_path):
    model = onnx.load(model_path)
    return model

# Get input and output type count from the ONNX model using protobuf information
def get_io_count(model):
    input_count = 0
    output_count = 0
    custom_op_node_exists = False

    print("Note: This C++ CustomOp template generator currently only supports models with one custom op node.\n________\n")
    for node in model.graph.node:
        # Find CustomOp node using domain
        if node.domain == "ai.onnx.contrib" or node.domain == "com.microsoft.extensions":
            assert not custom_op_node_exists, "Error: Found more than one custom op node. Exactly one is expected."
            custom_op_node_exists = True
            input_count = len(node.input)
            output_count = len(node.output)

    if not custom_op_node_exists:
        sys.exit("Error: No custom op node present in provided model")

    return input_count, output_count

# Add initial CustomOp code to C++ header file for CustomOp template
def create_hpp(customop_template_filepath, op, op_name, input_type_count, output_type_count):
    customop_template = r'''
    #include <stdio.h>
    #include "ocos.h"
    #include "string_utils.h"
    #include "ustring.h"

    void* {custom_op}::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {{
      return CreateKernelImpl(api, info);
    }};

    const char* {custom_op}::GetName() const {{ return "{op_name}"; }};

    size_t {custom_op}::GetInputTypeCount() const {{
      return {input_type_count};
    }};

    // Note: the following method is not complete and contains a temporary default return value.
    // Change return value to appropriate data type based on ONNXTensorElementDataType
    // mapping of TensorProto.DataType value
    ONNXTensorElementDataType {custom_op}::GetInputType(size_t /*index*/) const {{
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }};

    size_t {custom_op}::GetOutputTypeCount() const {{
      return {output_type_count};
    }};

    // Note: the following method is not complete and contains a temporary default return value.
    // Change return value to appropriate data type based on ONNXTensorElementDataType
    // mapping of TensorProto.DataType value
    ONNXTensorElementDataType {custom_op}::GetOutputType(size_t /*index*/) const {{
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }};
    '''

    hpp_file = customop_template.format(custom_op = op, op_name = op_name, input_type_count = input_type_count, output_type_count = output_type_count)

    # Write code to C++ template file
    with open(customop_template_filepath,'wt') as file:
        print(f"Added C++ CustomOp template code to output filepath: {customop_template_filepath}\n")
        file.write(hpp_file)

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Generate C++ template code for a new custom operator.",
    )

    parser.add_argument(
        "onnx_model_filepath",
        type=pathlib.Path,
        help="Path to ONNX model with CustomOp node.",
    )

    parser.add_argument(
        "customop_template_filepath",
        type=pathlib.Path,
        help="Output file path to add C++ template code file.",
    )

    parsed_args = parser.parse_args(args)
    return parsed_args

def main(args):
    args = parse_args(args)
    model = load_onnx_model(args.onnx_model_filepath)
    input_count, output_count = get_io_count(model)
    create_hpp(customop_template_filepath = args.customop_template_filepath, op = "CustomOp", op_name = "CustomOpName", input_type_count = input_count, output_type_count = output_count)
    return input_count, output_count

if __name__ == "__main__":
    main(sys.argv[1:])
