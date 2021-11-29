import os
import fire
import onnx
import numpy

from onnx import onnx_pb, save_tensor, numpy_helper
from ._ortapi2 import OrtPyFunction


class ORTExtCommands:
    def __init__(self, model='model.onnx', testdata_dir=None) -> None:
        self._model = model
        self._testdata_dir = testdata_dir

    def run(self, *args):
        """
        Run an onnx model with the arguments as its inputs
        """
        op_func = OrtPyFunction.from_model(self._model)
        np_args = [numpy.asarray(_x) for _x in args]
        for _idx, _sch in enumerate(op_func.inputs):
            if _sch.type.tensor_type.elem_type == onnx_pb.TensorProto.FLOAT:
                np_args[_idx] = np_args[_idx].astype(numpy.float32)

        print(op_func(*np_args))
        if self._testdata_dir:
            testdir = os.path.expanduser(self._testdata_dir)
            target_dir = os.path.join(testdir, 'test_data_set_0')
            os.makedirs(target_dir, exist_ok=True)
            for _idx, _x in enumerate(np_args):
                fn = os.path.join(target_dir, "input_{}.pb".format(_idx))
                save_tensor(numpy_helper.from_array(_x, op_func.inputs[_idx].name), fn)
            onnx.save_model(op_func.onnx_model, os.path.join(testdir, 'model.onnx'))

    def selfcheck(self, *args):
        print("The extensions loaded, status: OK.")


if __name__ == '__main__':
    fire.Fire(ORTExtCommands)
