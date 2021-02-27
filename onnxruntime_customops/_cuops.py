import onnx
from onnx import onnx_pb as onnx_proto
from ._ocos import default_opset_domain, get_library_path  # noqa


class SingleOpGraph:
    input_id, output_id = (0, 1)
    customop_schema = {
        # op_type: (input_list, output_list)
        'GPT2Tokenizer': ([onnx.helper.make_tensor_value_info('input_text', onnx_proto.TensorProto.STRING, [None])]
                          [onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT64, [None, None])]),
        'VectorToString': ([onnx.helper.make_tensor_value_info("token_ids", onnx.TensorProto.INT64, [None, None])],
                           [onnx.helper.make_tensor_value_info('text', onnx_proto.TensorProto.STRING, [None])])
    }

    @classmethod
    def get_next_id(cls):
        if not hasattr(cls, '_id_counter'):
            cls._id_counter = 0
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def build_singleop_graph(cls, op_type, *args, **kwargs):
        inputs = cls.customop_schema[op_type][cls.input_id]
        outputs = cls.customop_schema[op_type][cls.output_id]
        attrs = {}
        if op_type == "GPT2Tokenizer":
            attrs['vocab'] = open(kwargs.get('vocab_file'), 'rb').read()
            attrs['merges'] = open(kwargs.get('merges_file'), 'rb').read()
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
