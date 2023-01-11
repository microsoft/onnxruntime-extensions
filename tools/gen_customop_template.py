'''
Generates the C++ template code for a new custom operator.

Usage: $ python tools\gen_customop_template.py onnx_model_filepath
'''

import onnx
import sys

# Load and check ONNX model
model = onnx.load(sys.argv[1])
onnx.checker.check_model(model)

# Get input and output type count from the ONNX model using protobuf information
input_count = 0
output_count = 0

# Note: if there is no custom op node in the graph, GetInputTypeCount() and
# GetOutputTypeCount() will return 0.
for node in model.graph.node:
    # Find CustomOp node using domain
    if node.domain == "ai.onnx.contrib" or node.domain == "com.microsoft.extensions":
        input_count = len(node.input)
        output_count = len(node.output)

# Initial CustomOp code to be populated in C++ header file for CustomOp template
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

cpp_file = customop_template.format(custom_op = "CustomOp", op_name = "CustomOpName", input_type_count = input_count, output_type_count = output_count)

# Write code to C++ temmplate file
new_op = "tools/custom_op_template.hpp"
with open(new_op,'wt') as file:
    file.write(cpp_file)