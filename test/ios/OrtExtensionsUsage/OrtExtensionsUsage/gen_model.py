#!/usr/bin/env python3

import onnx
from onnx import TensorProto, helper

custom_ops_domain = "com.microsoft.extensions"

graph = helper.make_graph(
    [  # nodes
        helper.make_node("DecodeImage", ["image"], ["bgr_data"], "DecodeImage", domain=custom_ops_domain),
    ],
    "DecodeImage",  # name
    [  # inputs
        helper.make_tensor_value_info("image", TensorProto.UINT8, ["image_length"]),
    ],
    [  # outputs
        helper.make_tensor_value_info("bgr_data", TensorProto.UINT8, ["bgr_data_h", "bgr_data_w", 3]),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid(custom_ops_domain, 1)])
onnx.save(model, r"decode_image.onnx")
