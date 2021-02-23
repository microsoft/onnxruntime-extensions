import onnx
import onnxruntime as _ort
import _mytorch as mytorch # noqa
import onnxruntime_customops as _ocos


class ONNXTraceSession:
    activated_sessions = []

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.torch_ops = []

    def __enter__(self):
        self.activated_sessions.append(self)
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        last = self.activated_sessions[-1]
        del self.activated_sessions[-1:]
        return last

    @classmethod
    def get_active_session(cls):
        return cls.activated_sessions[0] if cls.activated_sessions else None

    def set_outputs(self, output_list):
        pass


class EagerOp:

    _ort_session_options = None
    _id_counter = 0

    @classmethod
    def get_next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def get_ort_session_options(cls):
        if cls._ort_session_options is None:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_ocos.get_library_path())
            cls._ort_session_options = so
        return cls._ort_session_options

    def __init__(self):
        self._onnx_model = None
        self.ort_session = None

    def create_from_customop(self, op_type, *args, **kwargs):
        # op_s = _ocos.Opdef.get_opdef(op_type)
        #
        # inputs = [onnx.helper.make_tensor_value_info("i_{}".format(self.get_next_id()),
        #                                              is_.dtype, []) for is_ in op_s.get_input_types()]
        # outputs = [onnx.helper.make_tensor_value_info("o_{}".format(self.get_next_id()),
        #                                               is_.dtype, []) for is_ in op_s.get_output_types()]
        # cuop = onnx.helper.make_node(op_type,
        #                              [i_.name for i_ in inputs],
        #                              [o_.name for o_ in outputs],
        #                              "{}_{}".format(op_type, EagerOp.get_next_id()),
        #                              domain=_ocos.default_opset_domain())
        # graph = onnx.helper.make_graph([cuop],
        #                                "og_{}_{}".format(op_type, EagerOp.get_next_id()),
        #                                inputs,
        #                                outputs)

        graph = _ocos.Opdef.build_singleop_graph(op_type, *args, **kwargs)
        model = onnx.helper.make_model(graph)
        model.opset_import.extend([
            onnx.helper.make_operatorsetid(_ocos.default_opset_domain(), 1)])
        self._bind(model)
        return self

    @property
    def onnx_model(self):
        return self._oxml

    def _bind(self, oxml):
        self.inputs = list(oxml.graph.input)
        self.output = list(oxml.graph.output)
        self._oxml = oxml
        return self

    def _ensure_ort_session(self):
        if self.ort_session is None:
            sess = _ort.InferenceSession(self.onnx_model, self.get_ort_session_options())
            self.ort_session = sess
        return self.ort_session

    @staticmethod
    def from_customop(op_type, *args, **kwargs):
        return EagerOp().create_from_customop(op_type, *args, **kwargs)

    @staticmethod
    def from_model(path_or_model, *args, **kwargs):
        return EagerOp()._bind(onnx.load_model(path_or_model) if isinstance(path_or_model, str) else path_or_model)

    def _argument_map(self, *args, **kwargs):
        idx = 0
        feed = {}
        for i_ in self.inputs:
            feed[i_.name] = args[idx]

        return feed

    def __call__(self, *args, **kwargs):
        self._ensure_ort_session()
        return self.ort_session.run(None, self._argument_map(*args, **kwargs))
