# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import warnings
from pathlib import Path
from ._ortcustomops import (  # noqa
    PyCustomOpDef, add_custom_op, PyCustomOpDefAttribute)


def get_library_path():
    pkg_dir = Path(__file__).parent
    return str(pkg_dir / "_ortcustomops.pyd")


class Opdef:

    _odlist = {}

    def __init__(self, op_type, func):
        self.op_type = op_type
        self.body = func
        self._id = id(self)

    @staticmethod
    def declare(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__'):
            raise RuntimeError("Unexpected arguments {}.".format(args))
            # return Opdef._create(args[0])
        return lambda f: Opdef._create(f, *args, **kwargs)

    @staticmethod
    def _create(func, *args, **kwargs):
        name = kwargs.get('op_type', None)
        op_type = name or func.__name__
        opdef = Opdef(op_type, func)
        od_id = id(opdef)

        # Tells python this object cannot be destroyed
        # because it is also stored in C++ container.
        Opdef._odlist[od_id] = opdef
        opdef._nativedef = PyCustomOpDef()
        opdef._nativedef.op_type = op_type
        opdef._nativedef.obj_id = od_id
        if 'atts' in kwargs:
            warnings.warn("onnxruntime C API does not support attributes yet, "
                          "attributes are ignored.")
            # opdef._nativedef.atts = kwargs['atts']

        # TODO: add handle more types and multiple inputs/outputs.
        # by default the op is single in/out
        inputs = kwargs.get('inputs', None)
        if inputs is None:
            inputs = [PyCustomOpDef.dt_float]
        opdef._nativedef.input_types = inputs
        outputs = kwargs.get('outputs', None)
        if outputs is None:
            outputs = [PyCustomOpDef.dt_float]
        opdef._nativedef.output_types = outputs
        add_custom_op(opdef._nativedef)
        return opdef

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)


def _on_pyop_invocation(k_id, feed):
    if k_id not in Opdef._odlist:
        raise RuntimeError(
            "Unable to find function id={}. "
            "Did you decorate the operator with @onnx_op?.".format(k_id))
    op_ = Opdef._odlist[k_id]
    rv = op_.body(*feed)
    if isinstance(rv, tuple):
        # Multiple outputs.
        res = []
        for r in rv:
            res.append(r.shape)
            res.append(r.flatten().tolist())
        res = tuple(res)
    else:
        res = (rv.shape, rv.flatten().tolist())
    return (k_id, ) + res


PyCustomOpDef.install_hooker(_on_pyop_invocation)
