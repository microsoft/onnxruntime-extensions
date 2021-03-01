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
        return [cls.io_def('input_text', onnx_proto.TensorProto.STRING, [None])]

    @classmethod
    def get_outputs(cls):
        return [cls.io_def("input_ids", onnx.TensorProto.INT64, [None, None])]


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
                attr_data[k_] = '\n'.join(k + "\t" + " ".join([str(i) for i in v]) for k, v in v_.items())
            else:
                attr_data[k_] = v_
        return attr_data

# TODO: list all custom operators schema here:
# ...
# ...


class SingleOpGraph:
    @classmethod
    def get_next_id(cls):
        if not hasattr(cls, '_id_counter'):
            cls._id_counter = 0
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def build_my_graph(cls, op_class, *args, **kwargs):
        # opcls = globals()[op_type], support the op_name here?
        op_type = op_class.op_type()
        inputs = op_class.get_inputs()
        outputs = op_class.get_outputs()
        attrs = op_class.serialize_attr(kwargs)
        cuop = onnx.helper.make_node(op_type,
                                     [i_.name for i_ in inputs],
                                     [o_.name for o_ in outputs],
                                     "{}_{}".format(op_type, cls.get_next_id()),
                                     **attrs,
                                     domain=default_opset_domain())
        graph = onnx.helper.make_graph([cuop],
                                       "og_{}_{}".format(op_type, cls.get_next_id()),
                                       inputs,
                                       outputs)
        return graph
