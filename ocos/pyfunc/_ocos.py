# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import numpy as np
from pathlib import Path
from ._ortcustomops import (PyCustomOpDef, add_custom_op)


def get_library_path():
    pkg_dir = Path(__file__).parent
    return str(pkg_dir / "_ortcustomops.pyd")


class OpdefList:
    odlist = []


class Opdef:

    def __init__(self, op_type, func):
        self.op_type = op_type
        self.body = func
        self._id = id(self)

    @staticmethod
    def declare(*args, **kwargs):
        if len(args) > 0 and hasattr(args[0], '__call__'):
            return Opdef._create(args[0])
        else:
            return lambda f: Opdef._create(f, *args, **kwargs)

    @staticmethod
    def _create(func, *args, **kwargs):
        name = kwargs.get('op_type', None)
        op_type = name or func.__name__
        opdef = Opdef(op_type, func)
        od_id = id(opdef)
        OpdefList.odlist.append(opdef)
        opdef._nativedef = PyCustomOpDef()
        opdef._nativedef.op_type = op_type
        opdef._nativedef.obj_id = od_id

        # TODO: add handle more types and multiple inputs/outputs.
        # by default the op is single in/out
        if kwargs.get('inputs', None) is None:
            opdef._nativedef.input_types = [PyCustomOpDef.dt_float]
        if kwargs.get('outputs', None) is None:
            opdef._nativedef.output_types = [PyCustomOpDef.dt_float]

        add_custom_op(opdef._nativedef)
        return opdef

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)


def _on_pyop_invocation(k_id, feed):
    for op_ in OpdefList.odlist:
        if op_._nativedef.obj_id == k_id:
            rv = op_.body(*feed)
            return k_id, rv.shape, rv.flatten().tolist()

    # return a dummy result if there is no function found,
    # an exception should be raised in C++ custom op implementation.
    fetch = np.ones([1, 1], np.float32)
    return 0, fetch.shape, fetch.flatten().tolist()


PyCustomOpDef.install_hooker(_on_pyop_invocation)
