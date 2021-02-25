import onnx
import onnxruntime as _ort

from ._ocos import default_opset_domain, get_library_path  # noqa
from ._onnx_ops import SingleOpTorch


class EagerOp:

    @classmethod
    def get_ort_session_options(cls):
        if not hasattr(cls, '_ort_session_options') or cls._ort_session_options is None:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(get_library_path())
            cls._ort_session_options = so
        return cls._ort_session_options

    def __init__(self):
        self._onnx_model = None
        self.ort_session = None

    def create_from_customop(self, op_type, *args, **kwargs):
        graph = SingleOpTorch.build_singleop_graph(op_type, *args, **kwargs)
        model = onnx.helper.make_model(graph)
        model.opset_import.extend([
            onnx.helper.make_operatorsetid(default_opset_domain(), 1)])
        self._bind(model)
        # onnx.save_model(model, "{}.onnx".format(op_type))
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
            sess = _ort.InferenceSession(self.onnx_model.SerializeToString(), self.get_ort_session_options())
            self.ort_session = sess

        return self.ort_session

    @classmethod
    def from_customop(cls, op_type, *args, **kwargs):
        return cls().create_from_customop(op_type, *args, **kwargs)

    @classmethod
    def from_model(cls, path_or_model, *args, **kwargs):
        return cls()._bind(onnx.load_model(path_or_model) if isinstance(path_or_model, str) else path_or_model)

    def _argument_map(self, *args, **kwargs):
        idx = 0
        feed = {}
        for i_ in self.inputs:
            feed[i_.name] = args[idx]

        return feed

    def __call__(self, *args, **kwargs):
        try:
            self._ensure_ort_session()
            return self.ort_session.run(None, self._argument_map(*args, **kwargs))
            # outputs = dict(zip([n_.name for n_ in self.ort_session.get_outputs()], outseq))

            # outputs = [Tensor.from_onnx(self.ort_session, outseq[n_], out_.name
            #                                   ) for n_, out_ in enumerate(self.ort_session.get_outputs())]
            # # FIXME: The tokenizer support attention_mask output
            # outputs.append(Tensor([1] * outputs[0].value.shape[1], 'attention_mask'))
        except Exception as e:
            print(e)
