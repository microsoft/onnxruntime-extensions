# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
cmd.py: cli commands for onnxruntime_extensions
"""

import os
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="ORT Extension commands")
    parser.add_argument("command", choices=["run", "selfcheck"])
    parser.add_argument("--model", default="model.onnx", help="Path to the ONNX model file")
    parser.add_argument("--testdata-dir", help="Path to the test data directory")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Additional arguments")

    args = parser.parse_args()

    ort_commands = ORTExtCommands(model=args.model, testdata_dir=args.testdata_dir)

    if args.command == "run":
        ort_commands.run(*args.args)
    elif args.command == "selfcheck":
        ort_commands.selfcheck(*args.args)


if __name__ == '__main__':
    main()
