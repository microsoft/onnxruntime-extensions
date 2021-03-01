import onnx
from onnx import onnx_pb as onnx_proto
from ._ocos import default_opset_domain, get_library_path  # noqa


class CustomOp:
    @classmethod
    def op_type(cls):
        return cls.__name__

    @classmethod
    def get_inputs(cls): return None

    @classmethod
    def get_output(cls): return None

    iodef = onnx.helper.make_tensor_value_info


class GPT2Tokenizer(CustomOp):
    @classmethod
    def get_inputs(cls):
        return [cls.iodef('input_text', onnx_proto.TensorProto.STRING, [None])]

    @classmethod
    def get_outputs(cls):
        return [cls.iodef("input_ids", onnx.TensorProto.INT64, [None, None])]


class VectorToString(CustomOp):
    @classmethod
    def get_inputs(cls):
        return [cls.iodef("token_ids", onnx.TensorProto.INT64, [None, None])]

    @classmethod
    def get_outputs(cls):
        return [cls.iodef('text', onnx_proto.TensorProto.STRING, [None])]


class SingleOpGraph:
    @classmethod
    def get_next_id(cls):
        if not hasattr(cls, '_id_counter'):
            cls._id_counter = 0
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def build_singleop_graph(cls, op_type, *args, **kwargs):
        opcls = globals()[op_type]
        inputs = opcls.get_inputs()
        outputs = opcls.get_outputs()
        cuop = onnx.helper.make_node(op_type,
                                     [i_.name for i_ in inputs],
                                     [o_.name for o_ in outputs],
                                     "{}_{}".format(op_type, cls.get_next_id()),
                                     **kwargs,
                                     domain=default_opset_domain())
        graph = onnx.helper.make_graph([cuop],
                                       "og_{}_{}".format(op_type, cls.get_next_id()),
                                       inputs,
                                       outputs)
        return graph
