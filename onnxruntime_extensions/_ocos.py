# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
"""
_ocos.py: PythonOp implementation
"""
import os
import sys
import glob


def _search_cuda_dir():
    paths = os.getenv("PATH", "").split(os.pathsep)
    for path in paths:
        for filename in glob.glob(os.path.join(path, "cudart64*.dll")):
            return os.path.dirname(filename)

    return None


if sys.platform == "win32":
    from . import _version  # noqa: E402

    if hasattr(_version, "cuda"):
        cuda_path = _search_cuda_dir()
        if cuda_path is None:
            raise RuntimeError("Cannot locate CUDA directory in the environment variable for GPU package")

        os.add_dll_directory(cuda_path)


from ._extensions_pydll import (  # noqa
    PyCustomOpDef,
    enable_py_op,
    add_custom_op,
    hash_64,
    default_opset_domain,
)


def get_library_path():
    """
    The custom operator library binary path
    :return: A string of this library path.
    """
    mod = sys.modules["onnxruntime_extensions._extensions_pydll"]
    return mod.__file__


class Opdef:
    _odlist = {}

    def __init__(self, op_type, func):
        self.op_type = op_type
        self.body = func
        self._id = id(self)

    @staticmethod
    def declare(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], "__call__"):
            raise RuntimeError("Unexpected arguments {}.".format(args))
            # return Opdef._create(args[0])
        return lambda f: Opdef.create(f, *args, **kwargs)

    @staticmethod
    def create(func, *args, **kwargs):
        name = kwargs.get("op_type", None)
        op_type = name or func.__name__
        opdef = Opdef(op_type, func)
        od_id = id(opdef)

        # Tells python this object cannot be destroyed
        # because it is also stored in C++ container.
        Opdef._odlist[od_id] = opdef
        opdef._nativedef = PyCustomOpDef()
        opdef._nativedef.op_type = op_type
        opdef._nativedef.obj_id = od_id

        inputs = kwargs.get("inputs", None)
        if inputs is None:
            inputs = [PyCustomOpDef.dt_float]
        opdef._nativedef.input_types = inputs
        outputs = kwargs.get("outputs", None)
        if outputs is None:
            outputs = [PyCustomOpDef.dt_float]
        opdef._nativedef.output_types = outputs
        attrs = kwargs.get("attrs", None)
        if attrs is None:
            attrs = {}
        elif isinstance(attrs, (list, tuple)):
            attrs = {k: PyCustomOpDef.dt_string for k in attrs}
        opdef._nativedef.attrs = attrs
        add_custom_op(opdef._nativedef)
        return opdef

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)

    def cast_attributes(self, attributes):
        res = {}
        for k, v in attributes.items():
            if self._nativedef.attrs[k] == PyCustomOpDef.dt_int64:
                res[k] = int(v)
            elif self._nativedef.attrs[k] == PyCustomOpDef.dt_float:
                res[k] = float(v)
            elif self._nativedef.attrs[k] == PyCustomOpDef.dt_string:
                res[k] = v
            else:
                raise RuntimeError("Unsupported attribute type {}.".format(self._nativedef.attrs[k]))
        return res


def _on_pyop_invocation(k_id, feed, attributes):
    if k_id not in Opdef._odlist:
        raise RuntimeError(
            "Unable to find function id={}. " "Did you decorate the operator with @onnx_op?.".format(k_id)
        )
    op_ = Opdef._odlist[k_id]
    rv = op_.body(*feed, **op_.cast_attributes(attributes))
    if isinstance(rv, tuple):
        # Multiple outputs.
        res = []
        for r in rv:
            res.append(r.shape)
            res.append(r.flatten().tolist())
        res = tuple(res)
    else:
        res = (rv.shape, rv.flatten().tolist())
    return (k_id,) + res


PyCustomOpDef.install_hooker(_on_pyop_invocation)
