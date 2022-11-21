# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import onnx
import numpy
from onnx import onnx_pb as onnx_proto
from ._ocos import default_opset_domain, Opdef, PyCustomOpDef


class CustomOp:

    @classmethod
    def op_type(cls):
        rcls = cls
        while CustomOp != rcls.__base__:
            rcls = rcls.__base__
        return rcls.__name__

    @classmethod
    def get_inputs(cls):
        return None

    @classmethod
    def get_outputs(cls):
        return None

    @classmethod
    def serialize_attr(cls, attrs):
        """
        Only support serialize the basic python type like list or dict,
        All other types needs to be serialized by the users
        :param attrs: the dict attributes
        :return: the dict of serialized data
        """
        return attrs

    io_def = onnx.helper.make_tensor_value_info


class GPT2Tokenizer(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def('input_text', onnx_proto.TensorProto.STRING, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def("input_ids", onnx.TensorProto.INT64, [None, None]),
            cls.io_def('attention_mask', onnx.TensorProto.INT64, [None, None])
        ]


class VectorToString(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [cls.io_def("token_ids", onnx.TensorProto.INT64, [])]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('text', onnx_proto.TensorProto.STRING, [None])]

    @classmethod
    def serialize_attr(cls, attrs):
        attr_data = {}
        for k_, v_ in attrs.items():
            if k_ == 'map' and isinstance(v_, dict):
                attr_data[k_] = '\n'.join(k + "\t" +
                                          " ".join([str(i) for i in v])
                                          for k, v in v_.items())
            elif k_ == 'map' and isinstance(v_, str):
                attr_data[k_] = v_
            else:
                attr_data[k_] = v_
        return attr_data


class StringMapping(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [cls.io_def("input", onnx.TensorProto.STRING, [])]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('output', onnx_proto.TensorProto.STRING, [])]

    @classmethod
    def serialize_attr(cls, attrs):
        attr_data = {}
        for k_, v_ in attrs.items():
            if k_ == 'map' and isinstance(v_, dict):
                attr_data[k_] = '\n'.join(k + "\t" + v for k, v in v_.items())
            elif k_ == 'map' and isinstance(v_, str):
                attr_data[k_] = v_
            else:
                attr_data[k_] = v_
        return attr_data


class MaskedFill(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def("value", onnx.TensorProto.STRING, [None]),
            cls.io_def("mask", onnx.TensorProto.BOOL, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('output', onnx_proto.TensorProto.STRING, [None])]


class StringToVector(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [cls.io_def("text", onnx.TensorProto.STRING, [None])]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('token_ids', onnx_proto.TensorProto.INT64, [])]

    @classmethod
    def serialize_attr(cls, attrs):
        attr_data = {}
        for k_, v_ in attrs.items():
            if k_ == 'map' and isinstance(v_, dict):
                attr_data[k_] = '\n'.join(k + "\t" +
                                          " ".join([str(i) for i in v])
                                          for k, v in v_.items())
            elif k_ == 'map' and isinstance(v_, str):
                attr_data[k_] = v_
            elif k_ == 'unk' and isinstance(v_, list):
                attr_data[k_] = ' '.join(str(i) for i in v_)
            else:
                attr_data[k_] = v_
        return attr_data


class BlingFireSentenceBreaker(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [cls.io_def("text", onnx.TensorProto.STRING, [None])]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('sentence', onnx_proto.TensorProto.STRING, [])]

    @classmethod
    def serialize_attr(cls, attrs):
        attrs_data = {}
        for k_, v_ in attrs.items():
            if k_ == 'model':
                with open(v_, "rb") as model_file:
                    attrs_data[k_] = model_file.read()
            else:
                attrs_data[k_] = v_
        return attrs_data


class SegmentExtraction(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [cls.io_def("input", onnx.TensorProto.INT64, [None, None])]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('position', onnx_proto.TensorProto.INT64, [None, 2]),
            cls.io_def('value', onnx_proto.TensorProto.INT64, [None])
        ]


class BertTokenizer(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [cls.io_def("text", onnx.TensorProto.STRING, [None])]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('input_ids', onnx_proto.TensorProto.INT64, [None]),
            cls.io_def('token_type_ids', onnx_proto.TensorProto.INT64, [None]),
            cls.io_def('attention_mask', onnx_proto.TensorProto.INT64, [None])
        ]

    @classmethod
    def serialize_attr(cls, attrs):
        attrs_data = {}
        for k_, v_ in attrs.items():
            if k_ == 'vocab_file':
                with open(v_, "r", encoding='utf-8') as model_file:
                    lines = model_file.readlines()
                    attrs_data[k_] = '\n'.join(lines)
            else:
                attrs_data[k_] = v_
        return attrs_data


class StringECMARegexReplace(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def("input", onnx.TensorProto.STRING, [None]),
            cls.io_def("pattern", onnx.TensorProto.STRING, [None]),
            cls.io_def("rewrite", onnx.TensorProto.STRING, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('output', onnx_proto.TensorProto.STRING, [None])]


class BertTokenizerDecoder(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def("ids", onnx.TensorProto.INT64, [None]),
            cls.io_def("position", onnx.TensorProto.INT64, [None, None])
        ]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('str', onnx_proto.TensorProto.STRING, [None])]

    @classmethod
    def serialize_attr(cls, attrs):
        attrs_data = {}
        for k_, v_ in attrs.items():
            if k_ == 'vocab_file':
                with open(v_, "r", encoding='utf-8') as model_file:
                    lines = model_file.readlines()
                    attrs_data[k_] = '\n'.join(lines)
            else:
                attrs_data[k_] = v_
        return attrs_data


class SentencepieceTokenizer(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def('inputs', onnx_proto.TensorProto.STRING, [None]),
            cls.io_def('nbest_size', onnx_proto.TensorProto.INT64, [None]),
            cls.io_def('alpha', onnx_proto.TensorProto.FLOAT, [None]),
            cls.io_def('add_bos', onnx_proto.TensorProto.BOOL, [None]),
            cls.io_def('add_eos', onnx_proto.TensorProto.BOOL, [None]),
            cls.io_def('reverse', onnx_proto.TensorProto.BOOL, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('tokens', onnx_proto.TensorProto.INT32, [None]),
            cls.io_def('indices', onnx_proto.TensorProto.INT64, [None])
        ]


class SentencepieceDecoder(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def("ids", onnx.TensorProto.INT64, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def('str', onnx_proto.TensorProto.STRING, [None])]


class Inverse(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def('input', onnx_proto.TensorProto.FLOAT, [None, None])
        ]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('output', onnx_proto.TensorProto.FLOAT, [None, None])
        ]


class ImageReader(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def('image_paths', onnx_proto.TensorProto.STRING, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('nchw_bytes', onnx_proto.TensorProto.UINT8, [None, None, None, None])
        ]


class GaussianBlur(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def('nhwc', onnx_proto.TensorProto.FLOAT, [None, None, None, None]),
            cls.io_def('kernel_size', onnx_proto.TensorProto.INT64, [None]),
            cls.io_def('sigma_xy', onnx_proto.TensorProto.DOUBLE, [None])
        ]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('gb_nhwc', onnx_proto.TensorProto.FLOAT, [None, None, None, None])
        ]


class ImageDecoder(CustomOp):

    @classmethod
    def get_inputs(cls):
        return [
            cls.io_def('raw_input_image', onnx_proto.TensorProto.UINT8, [])
        ]

    @classmethod
    def get_outputs(cls):
        return [
            cls.io_def('decoded_image', onnx_proto.TensorProto.UINT8, [None, None, 3])
        ]


class SingleOpGraph:

    @classmethod
    def get_next_id(cls):
        if not hasattr(cls, '_id_counter'):
            cls._id_counter = 0
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def build_my_graph(cls, op_class, *args, **kwargs):
        if isinstance(op_class, str):
            op_class = cls.get_op_class(op_class)

        op_type = op_class.op_type()
        inputs = op_class.get_inputs()
        outputs = op_class.get_outputs()
        attrs = op_class.serialize_attr(kwargs)
        cuop = onnx.helper.make_node(op_type, [i_.name for i_ in inputs],
                                     [o_.name for o_ in outputs],
                                     "{}_{}".format(op_type,
                                                    cls.get_next_id()),
                                     **attrs,
                                     domain=default_opset_domain())
        graph = onnx.helper.make_graph([cuop], "og_{}_{}".format(
            op_type, cls.get_next_id()), inputs, outputs)
        return graph

    @staticmethod
    def get_op_class(op_type):
        return globals()[op_type]


# TODO: have a C++ impl.
def _argsort_op(x, dim):
    d = numpy.argsort(x, dim)
    return d[:, ::-1]


Opdef.create(_argsort_op,
             op_type='ArgSort',
             inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int64],
             outputs=[PyCustomOpDef.dt_int64])
