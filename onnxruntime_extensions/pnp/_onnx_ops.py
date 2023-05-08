# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################
import warnings
import numpy as np
from onnx import helper, defs as onnx_defs, onnx_pb as onnx_proto
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE


DEFAULT_OPSET_NUMBER = 13  # The maximum opset supported by the converter in the code branch.
# From https://github.com/onnx/onnx/blob/master/docs/Versioning.md
OPSET_TO_IR_VERSION = {
    1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
    7: 3, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7,
    13: 7, 14: 7, 15: 8, 16: 8, 17: 8
}
if hasattr(helper, 'VERSION_TABLE'):
    OPSET_TO_IR_VERSION = {row[2]: row[1] for row in helper.VERSION_TABLE}


def _get_main_opset_version(model):
    """
    Returns the main opset version.
    """
    for op in model.opset_import:
        if op.domain == '' or op.domain == 'ai.onnx':
            return op.version
    return None


def onnx_builtin_opset_version():
    return onnx_defs.onnx_opset_version()


def get_maximum_opset_supported():
    return min(DEFAULT_OPSET_NUMBER, onnx_builtin_opset_version())


def make_model_ex(graph, imported_opset_pairs, target_default_opset, **kwargs):
    onnx_model = helper.make_model(graph, **kwargs)

    # Merge operator sets for the same domain, the largest version number would be kept
    purified_operator_set = dict()
    for op_domain, op_version in imported_opset_pairs:
        if op_domain not in purified_operator_set:
            if op_domain == '' or op_domain == 'ai.onnx':
                # Initializers are a subset of graph inputs for IR_VERSION <= 3 (target opset < 8).
                # Need upgrade opv since initializers are separate for IR_VERSION >= 4 to pass onnx.checker.
                if op_version < 8 and target_default_opset is not None and target_default_opset >= 8:
                    op_version = 8
            purified_operator_set[op_domain] = op_version
        else:
            purified_operator_set[op_domain] = max(purified_operator_set[op_domain], op_version)

    # Fill operator sets
    i = 0
    for op_domain, op_version in purified_operator_set.items():
        if i == 0 and len(onnx_model.opset_import) == 1:
            # Overwrite the default operator set created by helper.make_model(...)
            op_set = onnx_model.opset_import[0]
        else:
            # Just create one ONNX element in opset_import
            op_set = onnx_model.opset_import.add()
        op_set.domain = op_domain
        op_set.version = op_version
        i += 1
        if op_domain == '' or op_domain == 'ai.onnx':
            if target_default_opset < op_version:
                raise RuntimeError(('The specified opset %d is too low to convert this model, ' +
                                    'which requires at least opset %d.') % (target_default_opset, op_version))
            elif target_default_opset > op_version:
                warnings.warn('The maximum opset needed by this model is only %d.' % op_version)
            else:
                pass

    opv = _get_main_opset_version(onnx_model) or target_default_opset
    irv = OPSET_TO_IR_VERSION.get(opv, onnx_proto.IR_VERSION)
    onnx_model.ir_version = irv
    return onnx_model


class _ONNXModelOperator:
    def __init__(self, name, model, input, output):
        self.name = name
        self.model = model
        self.input = input
        self.output = output

    def __repr__(self):
        """
        without this method, it's too slow for the debugging.
        :return:
        """
        return "name: {}, input: {}, output: {}".format(self.name, self.input, self.output)

    @property
    def op_type(self):
        return 'ModelOp'


class ONNXElementContainer:

    opdict_counter = {}

    def __init__(self, target_opset, parent=None):
        """
        :param target_opset: number, for example, 7 for ONNX 1.2, and 8 for ONNX 1.3.
        """
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.value_info = []
        self.nodes = []
        self.node_domain_version_pair_sets = set()
        self.target_opset = target_opset
        self.enable_optimizer = True
        self.parent = parent

    # the following property make this container be compatible with onnx.GraphProto
    @property
    def initializer(self):
        return self.initializers

    @property
    def input(self):
        return self.inputs

    @property
    def output(self):
        return self.outputs

    @staticmethod
    def _make_value_info(variable):
        value_info = helper.ValueInfoProto()
        value_info.name = variable.full_name
        value_info.type.CopyFrom(variable.type.to_onnx_type())
        if variable.type.doc_string:
            value_info.doc_string = variable.type.doc_string
        return value_info

    def add_input(self, variable):
        """
        Add our Variable object defined _parser.py into the the input list of the final ONNX model

        :param variable: The Variable object to be added
        """
        self.inputs.append(self._make_value_info(variable))

    def add_output(self, variable):
        """1
        Add our Variable object defined _parser.py into the the output list of the final ONNX model

        :param variable: The Variable object to be added
        """
        self.outputs.append(self._make_value_info(variable))

    def add_initializer(self, name, onnx_type, shape, content):
        """
        Add a TensorProto into the initializer list of the final ONNX model

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        """
        if any(d is None for d in shape):
            raise ValueError('Shape of initializer cannot contain None')
        tensor = helper.make_tensor(name, onnx_type, shape, content)
        self.initializers.append(tensor)

    def add_value_info(self, variable):
        self.value_info.append(self._make_value_info(variable))

    def add_node(self, op_type, inputs, outputs, op_domain='', op_version=1, **attrs):
        """
        Add a NodeProto into the node list of the final ONNX model. If the input operator's domain-version information
        cannot be found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the type of the NodeProto
        :param inputs: A list of strings. They are the input variables' names of the considered NodeProto
        :param outputs: A list of strings. They are the output variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the operator we are trying to add.
        :param attrs: A Python dictionary. Keys and values are attributes' names and attributes' values, respectively.
        """

        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
        if not isinstance(inputs, (list, tuple)) or not all(isinstance(s, str) for s in inputs):
            type_list = ','.join(list(str(type(s)) for s in inputs))
            raise ValueError('Inputs must be a list of string but get [%s]' % type_list)
        if not isinstance(outputs, (list, tuple)) or not all(isinstance(s, str) for s in outputs):
            type_list = ','.join(list(str(type(s)) for s in outputs))
            raise ValueError('Outputs must be a list of string but get [%s]' % type_list)
        for k, v in attrs.items():
            if v is None:
                raise ValueError('Failed to create ONNX node. Undefined attribute pair (%s, %s) found' % (k, v))

        node = helper.make_node(op_type, inputs, outputs, **attrs)
        node.domain = op_domain

        self.node_domain_version_pair_sets.add((op_domain, op_version))
        self.nodes.append(node)

    def add_model_node(self, inputs, outputs, name, model):
        self.nodes.append(_ONNXModelOperator(name=name, model=model, input=inputs, output=outputs))

    @classmethod
    def get_unique_operator_name(cls, op_type: str):
        name = op_type.lower()
        nn = cls.opdict_counter.get(name, 0)
        cls.opdict_counter[name] = nn + 1
        return name if nn == 0 else "{}_{}".format(name, nn+1)


def _create_name_or_use_existing_one(container, op_type, name):
    return name or container.get_unique_operator_name(op_type)


class _OpSchema:
    _ox = None  # will be assigned by ONNXModelBuilder.

    def __init__(self, *args, **kwargs):
        # self.op_builder = None
        self.apply_fn = args[0]
        self.inputs = kwargs['inputs'] if 'inputs' in kwargs else []
        self.outputs = kwargs['outputs'] if 'outputs' in kwargs else []

    def __call__(self, *args, **kwargs):
        assert self._ox is not None, 'no builder instance was created'
        return self.apply_fn(self._ox, *args, **kwargs)

    # def __get__(self, instance, owner):
    #     if owner.__name__ == '_ONNXModelBuilder':
    #         self.op_builder = instance
    #     return self


def schema(apply_fn=None, *args, **kwargs):
    if apply_fn is None:
        def wrapper(fn):
            return _OpSchema(fn, *args, **kwargs)
        return wrapper
    else:
        # used as a function.
        return _OpSchema(apply_fn, *args, **kwargs)


class _ONNXOperatorAPI:
    _dt = onnx_proto.TensorProto
    def get_unique_tensor_name(self, base): pass  # implemented by the model builder

    def _apply_unary_operation(self, op_type, input_name, output_name, container, operator_name, **attrs):
        name = _create_name_or_use_existing_one(container, op_type, operator_name)
    
        attrs['name'] = name
        if container.target_opset < 6:
            attrs['consumed_inputs'] = [0]
            op_version = 1
        else:
            op_version = 6
    
        container.add_node(op_type, input_name, output_name, op_version=op_version, **attrs)
    
    def _apply_basic_numerical_operation(self, op_type, input_names, output_name, container, operator_name,
                                         axis, broadcast):
        name = _create_name_or_use_existing_one(container, op_type, operator_name)
    
        attrs = {}
        if container.target_opset < 7:
            # Before ONNX-1.2 (opset 7), broadcasting behavior is Caffe2-like.
            if axis is not None:
                attrs['axis'] = axis
            if broadcast is not None:
                attrs['broadcast'] = broadcast
    
            if container.target_opset < 6:
                attrs['consumed_inputs'] = [0, 0]
                op_version = 1
            else:
                op_version = 6
        else:
            # Since ONNX-1.2 (opset 7), broadcasting behavior is Numpy-like, so we don't need to specify any attributes
            op_version = 7
    
        container.add_node(op_type, input_names, output_name, op_version=op_version, name=name, **attrs)
    
    def _apply_pointwise_operation(self, op_type, input_names, output_name, container, operator_name):
        name = _create_name_or_use_existing_one(container, op_type, operator_name)
        attrs = {}
    
        if container.target_opset < 6:
            attrs['consumed_inputs'] = [0] * len(input_names)
            op_version = 1
        elif container.target_opset < 8:
            op_version = 6
        else:
            if container.target_opset < 12 or op_type == 'Mean':
                op_version = 8
            else:
                op_version = 12
    
        container.add_node(op_type, input_names, output_name, op_version=op_version, name=name, **attrs)
    
    def abs(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Abs', input_name, output_name, container, operator_name=operator_name)
        return output_name
    
    def add(self, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
        self._apply_basic_numerical_operation('Add', input_names, output_name, container, operator_name=operator_name,
                                         axis=axis, broadcast=broadcast)
        return output_name
    
    def argmax(self, input_name, output_name, container, operator_name=None, axis=0, keepdims=1,
                     select_last_index=0):
        name = _create_name_or_use_existing_one(container, 'ArgMax', operator_name)
        attrs = {'axis': axis, 'keepdims': keepdims}
        if container.target_opset < 11:
            op_version = 1
        elif container.target_opset < 12:
            op_version = 11
        else:
            op_version = 12
            attrs['select_last_index'] = select_last_index
        container.add_node('ArgMax', input_name, output_name, op_version=op_version, name=name, **attrs)
        return output_name
    
    def argmin(self, input_name, output_name, container, operator_name=None, axis=0, keepdims=1,
                     select_last_index=0):
        name = _create_name_or_use_existing_one(container, 'ArgMin', operator_name)
        attrs = {'axis': axis, 'keepdims': keepdims}
        if container.target_opset < 11:
            op_version = 1
        elif container.target_opset < 12:
            op_version = 11
        else:
            op_version = 12
            attrs['select_last_index'] = select_last_index
        container.add_node('ArgMin', input_name, output_name, op_version=op_version, name=name, **attrs)
        return output_name

    def affine(self, input_name, output_name, container, operator_name=None, alpha=1., beta=0.):
        if container.target_opset < 9:
            op_type = 'Affine'
            name = _create_name_or_use_existing_one(container, 'Affine', operator_name)
            attrs = {'name': name, 'alpha': alpha, 'beta': beta}
            container.add_node(op_type, input_name, output_name, **attrs)
        else:
            name = _create_name_or_use_existing_one(container, 'Affine', operator_name)
            # Define a and b.
            aName = self.get_unique_tensor_name(name + '_alpha')
            container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, [1], [alpha])
            bName = self.get_unique_tensor_name(name + '_beta')
            container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, [1], [beta])
    
            # Compute Z = a * X, where X is the original input.
            zName = self.get_unique_tensor_name(name + '_scaled')
            self.mul([aName, input_name], zName, container)
    
            # Compute Y = Z + b, where Y is the final output.
            self.add(self, [zName, bName], output_name, container)
        return output_name
    
    def batch_norm(self, input_names, output_names, container, operator_name=None,
                         epsilon=None, is_test=None, momentum=None, spatial=None):
        name = _create_name_or_use_existing_one(container, 'BatchNormalization', operator_name)
        attrs = {'name': name, 'epsilon': epsilon, 'momentum': momentum}
    
        if container.target_opset < 9:
            attrs['spatial'] = spatial
        if container.target_opset < 7:
            attrs['is_test'] = is_test
    
        if container.target_opset < 6:
            attrs['consumed_inputs'] = [0] * len(input_names)
            if len(input_names) > 3:
                attrs['consumed_inputs'][3] = 1
            if len(input_names) > 4:
                attrs['consumed_inputs'][4] = 2
            op_version = 1
        elif container.target_opset < 7:
            op_version = 6
        elif container.target_opset < 9:
            op_version = 7
        else:
            op_version = 9
    
        container.add_node('BatchNormalization', input_names, output_names, op_version=op_version, **attrs)
        return output_names
    
    def cast(self, input_name, output_name, container, operator_name=None, to=None):
        """
        :param to: enum defined in ONNX TensorProto.DataType, for example, TensorProto.FLOAT and TensorProto.INT64.
        """
        name = _create_name_or_use_existing_one(container, 'Cast', operator_name)
        attrs = {'name': name}
    
        d = onnx_proto.TensorProto.DataType.DESCRIPTOR
        allowed_type_name_and_type_enum_pairs = {v.number: k for k, v in d.values_by_name.items()}
        if to not in allowed_type_name_and_type_enum_pairs:
            raise ValueError('Attribute "to" must be one of %s' % allowed_type_name_and_type_enum_pairs.keys())
    
        if container.target_opset < 9:
            if to in [onnx_proto.TensorProto.STRING, onnx_proto.TensorProto.COMPLEX64, onnx_proto.TensorProto.COMPLEX128]:
                raise ValueError('Attribute "to" cannot correspond to a String or Complex TensorProto type.')
    
            if container.target_opset < 6:
                # Convert enum to string, for example, TensorProto.INT64 to 'INT64'
                attrs['to'] = allowed_type_name_and_type_enum_pairs[to]
                op_version = 1
            else:
                # Enum, for example, TensorProto.INT64
                attrs['to'] = to
                op_version = 6
        else:
            # Enum value, for example, TensorProto.INT64
            # String casting is supported in opset 9
            if to in [onnx_proto.TensorProto.COMPLEX64, onnx_proto.TensorProto.COMPLEX128]:
                raise ValueError('Attribute "to" cannot correspond to a Complex TensorProto type.')
            attrs['to'] = to
            op_version = 9
    
        container.add_node('Cast', input_name, output_name, op_version=op_version, **attrs)
        return output_name

    def clip(self, input_name, output_name, container, operator_name=None, max=None, min=None):
        name = _create_name_or_use_existing_one(container, 'Clip', operator_name)
        attrs = {'name': name}
    
        if container.target_opset < 11:
            if max is not None:
                attrs['max'] = float(max)
            if min is not None:
                attrs['min'] = float(min)
    
            if container.target_opset < 6:
                attrs['consumed_inputs'] = [0]
                op_version = 1
            else:
                op_version = 6
    
            container.add_node('Clip', input_name, output_name, op_version=op_version, **attrs)
        else:
            if container.target_opset < 12:
                op_version = 11
            else:
                op_version = 12
            if min is None and max is not None:
                raise RuntimeError("Operator 'Clip': min must be specified if max is.")
            inputs = [input_name]
    
            if min is not None:
                if isinstance(min, (np.ndarray, float, int)):
                    # add initializer
                    if isinstance(min, np.ndarray):
                        if len(min.shape) == 0:
                            min = [min]
                        elif min.shape == (1,):
                            min = list(min[0]) if hasattr(min[0], '__iter__') else list(min)
                        else:
                            raise RuntimeError("min must be an array of one element.")
                    else:
                        min = [min]
    
                    # container in sklearn-onnx stores the computation type in
                    # container.dtype.
                    min_name = self.get_unique_tensor_name('clip_min')
                    if op_version < 12:
                        min = np.array(min, dtype=getattr(container, 'dtype', np.float32))
                        container.add_initializer(min_name, getattr(container, 'proto_dtype',
                                                                    onnx_proto.TensorProto.FLOAT), [], [min[0]])
                    else:
                        min = np.array(min)
                        container.add_initializer(min_name, NP_TYPE_TO_TENSOR_TYPE[min.dtype], [], [min[0]])
                    min = min_name
                if isinstance(min, str):
                    inputs.append(min)
                else:
                    raise RuntimeError("Parameter 'min' must be a string or a float.")
    
            if max is not None:
                if min is None:
                    raise RuntimeError("Parameter 'min' must be specified if 'max' is.")
                if isinstance(max, (np.ndarray, float, int)):
                    # add initializer
                    if isinstance(max, np.ndarray):
                        if len(max.shape) == 0:
                            max = [max]
                        elif max.shape == (1,):
                            max = list(max[0]) if hasattr(max[0], '__iter__') else list(max)
                        else:
                            raise RuntimeError("max must be an array of one element.")
                    else:
                        max = [max]
    
                    max_name = self.get_unique_tensor_name('clip_max')
                    if op_version < 12:
                        max = np.array(max, dtype=getattr(container, 'dtype', np.float32))
                        container.add_initializer(max_name, getattr(container, 'proto_dtype',
                                                                    onnx_proto.TensorProto.FLOAT), [], [max[0]])
                    else:
                        max = np.array(max)
                        container.add_initializer(max_name, NP_TYPE_TO_TENSOR_TYPE[max.dtype], [], [max[0]])
                    max = max_name
                if isinstance(max, str):
                    inputs.append(max)
                else:
                    raise RuntimeError("Parameter 'max' must be a string or a float.")
    
            container.add_node('Clip', inputs, output_name, op_version=op_version,
                               **attrs)
        return output_name

    def concat(self, input_names, output_name, container, operator_name=None, axis=0):
        name = _create_name_or_use_existing_one(container, 'Concat', operator_name)
    
        if container.target_opset < 4:
            op_version = 1
        elif container.target_opset < 11:
            op_version = 4
        else:
            op_version = 11
    
        container.add_node('Concat', input_names, output_name, op_version=op_version, name=name, axis=axis)
        return output_name

    def concat_from_sequence(self, input_names, output_name, container, operator_name=None, axis=0, new_axis=None):
        name = _create_name_or_use_existing_one(container, 'Concat', operator_name)
        attrs = {'axis': axis}
        if new_axis is not None:
            attrs['new_axis'] = new_axis
        container.add_node('ConcatFromSequence', input_names, output_name, op_version=11, name=name, **attrs)
        return output_name

    def constant(self, input_names, output_name, container, operator_name=None, value=None):
        assert len(input_names) == 0  # only a placeholder to standardize the argument list.
        name = _create_name_or_use_existing_one(container, 'Constant', operator_name)
    
        if value is None:
            raise ValueError('Attribute "value" is a required argument.')
    
        if container.target_opset < 9:
            op_version = 1
        elif container.target_opset < 11:
            op_version = 9
        elif container.target_opset < 12:
            op_version = 11
        else:
            op_version = 12
    
        if op_version < 12:
            attrs = {'name': name, 'value': value}
        else:
            if isinstance(value, float):
                attrs = {'name': name, 'value_float': value}
            elif isinstance(value, int):
                attrs = {'name': name, 'value_int': value}
            elif isinstance(value, str):
                attrs = {'name': name, 'value_string': value}
            else:
                attrs = {'name': name, 'value': value}
    
        container.add_node('Constant', [], output_name, op_version=op_version, **attrs)
        return output_name

    def constant_of_shape(self, input_names, output_name, container, operator_name=None, value=None):
        attrs = {}
        if value is not None:
            attrs['value'] = value
        name = _create_name_or_use_existing_one(container, 'ConstantOfShape', operator_name)
        container.add_node('ConstantOfShape', input_names, output_name, name=name, op_version=9, **attrs)
        return output_name

    def conv(self, input_names, output_name, container, operator_name=None, **attrs):
        name = _create_name_or_use_existing_one(container, 'Conv', operator_name)
    
        if container.target_opset < 11:
            op_version = 1
        else:
            op_version = 11
    
        container.add_node('Conv', input_names, output_name, name=name, op_version=op_version, **attrs)
        return output_name

    def crop_height_width(self, input_name, output_name, container, operator_name=None,
                                top_border=0, bottom_border=0, left_border=0, right_border=0):
        name = container.get_unique_operator_name('CropHeightWidth')
        if container.target_opset < 9:
            # If operator set < 9, we can use the experimental Crop in ONNX.
            attrs = {'name': name, 'border': [left_border, top_border, right_border, bottom_border]}
            container.add_node('Crop', input_name, output_name, **attrs)
        else:
            # The experimental Crop in ONNX is removed after operator set 9, so we
            # switch to ONNX DynamicSlice operator.
    
            # CoreML only crops H- and W-axes.
            axes = [2, 3]
            axes_name = self.get_unique_tensor_name(name + '_axes')
            container.add_initializer(axes_name, onnx_proto.TensorProto.INT64,
                                      [len(axes)], axes)
    
            # Number of cropped pixels is the starting index of the remained region.
            starts = [top_border, left_border]
            starts_name = self.get_unique_tensor_name(name + '_starts')
            container.add_initializer(starts_name, onnx_proto.TensorProto.INT64,
                                      [len(starts)], starts)
    
            # First we assume no cropping is needed at the end of those axes.
            # We will change this right below depending on Crop's configuration.
            ends = [np.iinfo(np.int64).max] * 2
    
            # Crop n pixel means the end index (exclusive) is -n. Note that indexing
            # system is zero-based.
            if bottom_border > 0:
                ends[0] = -bottom_border
            if right_border > 0:
                ends[1] = -right_border
    
            # Add the adjusted ends.
            ends_name = self.get_unique_tensor_name(name + '_ends')
            container.add_initializer(ends_name, onnx_proto.TensorProto.INT64,
                                      [len(ends)], ends)
    
            # Collect all input names as a list because DynamicSlice has multiple inputs.
            input_list = [input_name, starts_name, ends_name, axes_name]
            container.add_node('DynamicSlice', input_list, output_name, op_version=9)
        return output_name

    def cumsum(self, input_names, output_names, container, operator_name=None, axis=None):
        name = _create_name_or_use_existing_one(container, 'cumsum', operator_name)
        assert axis is not None, "Axis in Op CumSum must be provided."
        axis_name = self.get_unique_tensor_name(name+'_dim')
        container.add_initializer(axis_name,
                                  onnx_proto.TensorProto.INT64,
                                  [1], [axis])
        container.add_node('CumSum', input_names + [axis_name], output_names, op_version=11, name=name)
        return output_names

    def div(self, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
        self._apply_basic_numerical_operation('Div', input_names, output_name,
                                              container, operator_name,
                                              axis, broadcast)
        return output_name

    def elu(self, input_name, output_name, container, operator_name=None, alpha=1.0):
        self._apply_unary_operation('Elu', input_name, output_name, container, operator_name, alpha=alpha)
        return output_name

    def equal(self, input_names, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'equal', operator_name)
        if container.target_opset < 7:
            op_version = 1
        elif container.target_opset < 9:
            op_version = 7
        else:
            op_version = 9
        container.add_node('Equal', input_names, output_name, name=name, op_version=op_version)
        return output_name
    
    def exp(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Exp', input_name, output_name, container, operator_name=operator_name)
        return output_name

    def floor(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Floor', input_name, output_name, container, operator_name=operator_name)
        return output_name

    def flatten(self, input_name, output_name, container, operator_name=None, axis=1):
        name = _create_name_or_use_existing_one(container, 'Flatten', operator_name)
        if container.target_opset < 9:
            op_version = 1
        elif container.target_opset < 11:
            op_version = 9
        else:
            op_version = 11
        container.add_node('Flatten', input_name, output_name, name=name, op_version=op_version, axis=axis)
        return output_name
    
    def gather(self, input_names, output_name, container, operator_name=None, axis=0):
        name = _create_name_or_use_existing_one(container, 'Gather', operator_name)
        if container.target_opset < 11:
            op_version = 1
        else:
            op_version = 11
    
        container.add_node('Gather', input_names, output_name, name=name, op_version=op_version, axis=axis)
        return output_name

    def gemm(self, input_name, output_name, container, operator_name=None, alpha=1.0, beta=1.0,
                   transA=0, transB=0):
        """
        Applies operator `gemm <https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm>`.
        """
        name = _create_name_or_use_existing_one(container, 'Gemm', operator_name)
        attrs = {'alpha': alpha, 'beta': beta, 'transA': transA, 'transB': transB}
        if container.target_opset < 5:
            attrs['op_version'] = 1
            attrs['broadcast'] = 1
        elif container.target_opset < 7:
            attrs['op_version'] = 6
            attrs['broadcast'] = 1
        elif container.target_opset < 11:
            attrs['op_version'] = 7
        else:
            attrs['op_version'] = 11
    
        container.add_node('Gemm', input_name, output_name, name=name, **attrs)
        return output_name

    @schema(outputs=((_dt.BOOL, []), ),)
    def greater(self, input_names, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Greater', operator_name)
        if container.target_opset < 7:
            op_version = 1
        elif container.target_opset < 9:
            op_version = 7
        else:
            op_version = 9
    
        container.add_node('Greater', input_names, output_name, name=name, op_version=op_version)
        return output_name

    def _apply_convert_compare_equal(self, input_names, output_name, container, operator_name,
                                     tf_op_string, onnx_op_string_rev, onnx_op_string):
        if container.target_opset < 7:
            raise ValueError(tf_op_string + " op is not supported for opset < 7")
        elif container.target_opset < 9:
            op_version = 7
        elif container.target_opset < 12:
            op_version = 9
        else:
            op_version = 12
        name = _create_name_or_use_existing_one(container, tf_op_string, operator_name)
        if op_version < 9:
            compare_input_0 = self.get_unique_tensor_name(name + '_input_0_cast')
            container.add_node('Cast', [input_names[0]], compare_input_0, name=name + '_input_0_cast', to=1)
            compare_input_1 = self.get_unique_tensor_name(name + '_input_1_cast')
            container.add_node('Cast', [input_names[1]], compare_input_1, name=name + '_input_1_cast', to=1)
            less_out = self.get_unique_tensor_name(name + '_less_out')
            container.add_node(onnx_op_string_rev, [compare_input_0, compare_input_1], less_out,
                               name=name + '_' + onnx_op_string_rev.lower(),
                               op_version=op_version)
            container.add_node('Not', less_out, output_name, name=name + '_not')
        elif op_version < 12:
            compare_node = self.get_unique_tensor_name(name + '_compare_node')
            container.add_node(onnx_op_string_rev, input_names, compare_node,
                               name=name + '_' + onnx_op_string_rev.lower(),
                               op_version=op_version)
            container.add_node('Not', [compare_node], output_name, name=name)
        else:
            container.add_node(onnx_op_string, input_names, output_name,
                               name=name + '_' + onnx_op_string_rev.lower(), op_version=op_version)

    def greater_or_equal(self, input_names, output_name, container, operator_name=None):
        self._apply_convert_compare_equal(input_names, output_name, container, operator_name,
                                          'GreaterEqual', 'Less', 'GreaterOrEqual')
        return output_name

    def less_or_equal(self, input_names, output_name, container, operator_name=None):
        self._apply_convert_compare_equal(input_names, output_name, container,
                                          operator_name, 'LessEqual', 'Greater', 'LessOrEqual')
        return output_name

    def gru(self, input_names, output_names, container, operator_name=None, output_seq=0, reset_after=0, **attrs):
        name = _create_name_or_use_existing_one(container, 'GRU', operator_name)
        if container.target_opset < 3:
            op_version = 1
            attrs['output_sequence'] = 1 if output_seq else 0
        else:
            attrs['linear_before_reset'] = 1 if reset_after else 0
            if container.target_opset <= 5:
                attrs['output_sequence'] = 1 if output_seq else 0
                op_version = 3
            else:
                op_version = 7
    
        container.add_node('GRU', input_names, output_names, name=name, op_version=op_version, **attrs)
        return output_names
    
    def hard_sigmoid(self, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
        self._apply_unary_operation('HardSigmoid', input_name, output_name, container, operator_name,
                               alpha=alpha, beta=beta)
        return output_name

    def identity(self, input_name, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Identity', operator_name)
        container.add_node('Identity', input_name, output_name, name=name)
        return output_name
    
    def instance_norm(self, input_names, output_name, container, operator_name=None, epsilon=1e-5):
        name = _create_name_or_use_existing_one(container, 'InstanceNormalization', operator_name)
        attrs = {'name': name, 'epsilon': epsilon}
    
        if container.target_opset < 2:
            attrs['consumed_inputs'] = [0] * len(input_names)
            op_version = 1
        else:
            op_version = 6
    
        container.add_node('InstanceNormalization', input_names, output_name, op_version=op_version, **attrs)
        return output_name

    def leaky_relu(self, input_name, output_name, container, operator_name=None, alpha=0.01):
        self._apply_unary_operation('LeakyRelu', input_name, output_name, container, operator_name, alpha=alpha)
        return output_name

    def less(self, input_names, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Less', operator_name)
        if container.target_opset < 7:
            op_version = 1
        elif container.target_opset < 9:
            op_version = 7
        else:
            op_version = 9
    
        container.add_node('Less', input_names, output_name, name=name, op_version=op_version)
        return output_name

    def log(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Log', input_name, output_name, container, operator_name=operator_name)
        return output_name

    def lstm(self, input_names, output_names, container, operator_name=None, output_seq=0, **attrs):
        name = _create_name_or_use_existing_one(container, 'LSTM', operator_name)
        if container.target_opset <= 6:
            attrs['output_sequence'] = 1 if output_seq else 0
            op_version = 1
        else:
            op_version = 7
        container.add_node('LSTM', input_names, output_names, name=name, op_version=op_version, **attrs)
        return output_names

    def matmul(self, input_names, output_name, container, operator_name=None):
        op_type = 'MatMul'
        name = _create_name_or_use_existing_one(container, op_type, operator_name)
        if container.target_opset <= 9:
            op_version = 1
        else:
            op_version = 9
        container.add_node(op_type, input_names, output_name, op_version=op_version, name=name)
        return output_name
    
    def max(self, input_names, output_name, container, operator_name=None):
        self._apply_pointwise_operation('Max', input_names, output_name, container, operator_name)
        return output_name
    
    def mean(self, input_names, output_name, container, operator_name=None):
        self._apply_pointwise_operation('Mean', input_names, output_name, container, operator_name)
        return output_name
    
    def min(self, input_names, output_name, container, operator_name=None):
        self._apply_pointwise_operation('Min', input_names, output_name, container, operator_name)
        return output_name

    def mul(self, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
        self._apply_basic_numerical_operation('Mul', input_names, output_name,
                                              container, operator_name=operator_name,
                                              axis=axis, broadcast=broadcast)
        return output_name

    def neg(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Neg', input_name, output_name, container, operator_name)
        return output_name
    
    def lpnormalization(self, input_name, output_name, container, operator_name=None, axis=1, p=2):
        name = _create_name_or_use_existing_one(container, 'LpNormalization', operator_name)
        container.add_node('LpNormalization', input_name, output_name, name=name, p=p, axis=axis)
        return output_name
    
    def not_op(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Not', input_name, output_name, container, operator_name)
        return output_name

    def or_op(self, input_names, output_names, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'or', operator_name)
        container.add_node('Or', input_names, output_names, op_version=7, name=name)
        return output_names
 
    def pad(self, input_name, output_name, container, operator_name=None, mode=None, pads=None, value=None,
                  onnx_type=onnx_proto.TensorProto.FLOAT):
        name = _create_name_or_use_existing_one(container, 'Pad', operator_name)
        attrs = {'name': name}
        inputs = input_name if isinstance(input_name, list) else [input_name]
    
        if mode is not None:
            attrs['mode'] = mode
    
        if container.target_opset < 11:
            if isinstance(pads, str):
                raise ValueError("Dynamic pad is not supported for opset < 11.")
            if value is not None:
                attrs['value'] = value
            if container.target_opset < 2:
                attrs['paddings'] = pads
                op_version = 1
            else:
                attrs['pads'] = pads
                op_version = 2
        else:
            op_version = 11
            if isinstance(pads, str):
                inputs.append(pads)
            else:
                pads_name = self.get_unique_tensor_name(name + '_pads')
                container.add_initializer(pads_name, onnx_proto.TensorProto.INT64, [len(pads)], pads)
                inputs.append(pads_name)
            if value is not None:
                value_name = self.get_unique_tensor_name(name + '_value')
                container.add_initializer(value_name, onnx_type, [], [value])
                inputs.append(value_name)
    
        container.add_node('Pad', inputs, output_name, op_version=op_version, **attrs)
        return output_name

    def parametric_softplus(self, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
        if alpha is None:
            alpha = [1.0]
        if beta is None:
            beta = [0.]
    
        name = _create_name_or_use_existing_one(container, 'ParametricSoftplus', operator_name)
        if container.target_opset < 9:
            if len(alpha) != 1 or len(beta) != 1:
                raise ValueError('alpha and beta must be 1-element lists')
            op_type = 'ParametricSoftplus'
            attrs = {'name': name, 'alpha': alpha[0], 'beta': beta[0]}
            container.add_node(op_type, input_name, output_name, **attrs)
        else:
            # Define three scalars: a, b, 1.
            aName = self.get_unique_tensor_name(name + '_alpha')
            aShape = [len(alpha)] if len(alpha) == 1 else [len(alpha), 1, 1]
            container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, aShape, alpha)
            bShape = [len(beta)] if len(beta) == 1 else [len(beta), 1, 1]
            bName = self.get_unique_tensor_name(name + '_beta')
            container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, bShape, beta)
            oneName = self.get_unique_tensor_name(name + '_one')
            container.add_initializer(oneName, onnx_proto.TensorProto.FLOAT, [1], [1.])
    
            # c = b * x
            cName = self.get_unique_tensor_name(name + '_c')
            self.mul([input_name, bName], cName, container)
    
            # d = exp(c)
            dName = self.get_unique_tensor_name(name + '_d')
            self.exp(cName, dName, container)
    
            # e = 1 + d
            eName = self.get_unique_tensor_name(name + '_e')
            self.add([dName, oneName], eName, container)
    
            # f = log(e)
            fName = self.get_unique_tensor_name(name + '_f')
            self.log(eName, fName, container)
    
            # g = a * f
            self.mul([fName, aName], output_name, container)
        return output_name

    def pow(self, input_names, output_name, container, operator_name=None, axis=None, broadcast=None):
        name = _create_name_or_use_existing_one(container, 'Pow', operator_name)
    
        attrs = {'name': name}
        if container.target_opset < 7:
            # Before ONNX-1.2, broadcasting behavior is Caffe2-like.
            if axis is not None:
                attrs['axis'] = axis
            if broadcast is not None:
                attrs['broadcast'] = broadcast
            op_version = 1
        elif container.target_opset < 12:
            # Since ONNX-1.2, broadcasting behavior is Numpy-like, so we don't need to specify any attributes
            op_version = 7
        else:
            op_version = 12
    
        container.add_node('Pow', input_names, output_name, op_version=op_version, **attrs)
        return output_name

    def prelu(self, input_name, output_name, container, operator_name=None, slp_rate=None):
        name = _create_name_or_use_existing_one(container, 'PRelu', operator_name)
        slp_rate_tensor_name = self.get_unique_tensor_name('slp_rate')
        s_shape = slp_rate.shape
        if container.target_opset < 7:
            s_shape = [len(slp_rate.flatten())]
        container.add_initializer(slp_rate_tensor_name, onnx_proto.TensorProto.FLOAT, s_shape, slp_rate.flatten())
    
        if container.target_opset < 6:
            container.add_node('PRelu', [input_name, slp_rate_tensor_name], output_name, op_version=1, name=name,
                               consumed_inputs=[0, 0])
        else:
            if container.target_opset < 7:
                op_version = 6
            elif container.target_opset < 9:
                op_version = 7
            else:
                # opset 9 supports unidirectional broadcasting
                op_version = 9
    
            container.add_node('PRelu', [input_name, slp_rate_tensor_name], output_name, op_version=op_version, name=name)
        return output_name

    def range(self, input_name, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Range', operator_name)
        container.add_node('Range', input_name, output_name, op_version=11, name=name)
        return output_name

    def reciprocal(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Reciprocal', input_name, output_name, container, operator_name=operator_name)
        return output_name

    # Some old ORT supports axis < 0 case, so put rank=0 as default.
    def reducesum(self, input_name, output_name, container, operator_name=None, axes=None, keepdims=1, rank=0):
        name = _create_name_or_use_existing_one(container, 'ReduceSum', operator_name)
        if axes is None:
            axes = []
        if container.target_opset < 13:
            if container.target_opset < 11:
                op_version = 1
                axes = [axis if axis >= 0 else axis + rank for axis in axes]
            else:
                op_version = 11
            container.add_node('ReduceSum', input_name, output_name, name=name,
                               op_version=op_version, axes=axes, keepdims=keepdims)
        else:
            if not isinstance(input_name, list):
                input_name = [input_name]
            op_version = 13
            if isinstance(axes, str):
                container.add_node('ReduceSum', input_name + [axes], output_name,
                                   op_version=op_version, name=name, keepdims=keepdims)
            elif axes is None or len(axes) == 0:
                container.add_node('ReduceSum', input_name, output_name,
                                   op_version=op_version, name=name, keepdims=keepdims)
            else:
                axes_name = self.get_unique_tensor_name(name + '_reducesum')
                container.add_initializer(axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes)
                container.add_node('ReduceSum', input_name + [axes_name], output_name,
                                   op_version=op_version, name=name, keepdims=keepdims)
        return output_name

    def reducemin(self, input_name, output_name, container, operator_name=None, axes=None, keepdims=1, rank=0):
        name = _create_name_or_use_existing_one(container, 'ReduceMin', operator_name)
        if axes is None:
            axes = []
        if container.target_opset < 13:
            if container.target_opset < 11:
                op_version = 1
                axes = [axis if axis >= 0 else axis + rank for axis in axes]
            else:
                op_version = 11
            container.add_node('ReduceMin', input_name, output_name, name=name,
                               op_version=op_version, axes=axes, keepdims=keepdims)
        else:
            if not isinstance(input_name, list):
                input_name = [input_name]
            op_version = 13
            if isinstance(axes, str):
                container.add_node('ReduceMin', input_name + [axes], output_name,
                                   op_version=op_version, name=name, keepdims=keepdims)
            elif axes is None or len(axes) == 0:
                container.add_node('ReduceMin', input_name, output_name,
                                   op_version=op_version, name=name, keepdims=keepdims)
            else:
                axes_name = self.get_unique_tensor_name(name + '_reducemin')
                container.add_initializer(axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes)
                container.add_node('ReduceMin', input_name + [axes_name], output_name,
                                   op_version=op_version, name=name, keepdims=keepdims)
        return output_name

    def relu(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Relu', input_name, output_name, container, operator_name)
        return output_name

    def relu_6(self, input_name, output_name, container, operator_name=None, zero_value=0.0):
        name_relu = _create_name_or_use_existing_one(container, 'relu', operator_name)
        name_relu_op = _create_name_or_use_existing_one(container, 'relu6', operator_name)
        self.relu(input_name, name_relu, container, name_relu_op+'_relu')
        self.clip(name_relu, output_name, container, name_relu_op + '_clip', zero_value+6, zero_value)

    def reshape(self, input_name, output_name, container, operator_name=None, desired_shape=None):
        if not isinstance(desired_shape, str) and len(list(i for i in desired_shape if i is not None and i < 0)) > 1:
            raise ValueError('There can only be one -1 in the targeted shape of a Reshape but got %s' % desired_shape)
    
        name = _create_name_or_use_existing_one(container, 'Reshape', operator_name)
    
        if container.target_opset < 5:
            container.add_node('Reshape', input_name, output_name, op_version=1, name=name, shape=desired_shape,
                               consumed_inputs=[0])
        else:
            if isinstance(desired_shape, str):
                desired_shape_name = desired_shape
            else:
                desired_shape_name = self.get_unique_tensor_name('shape_tensor')
                container.add_initializer(desired_shape_name, onnx_proto.TensorProto.INT64, [len(desired_shape)],
                                          desired_shape)
    
            # Create ONNX Reshape operator
            if isinstance(input_name, list):
                input_name.append(desired_shape_name)
            else:
                input_name = [input_name, desired_shape_name]
            container.add_node('Reshape', input_name, output_name, op_version=5, name=name)
        return output_name

    def resize(self, input_name, output_name, container, operator_name=None, mode='nearest',
                     coordinate_transformation_mode='asymmetric', scales=None):
        """
        :param mode: "nearest" or "linear"
        :param scales: a float tensor for scaling (upsampling or downsampling) all input dimensions
        """
        name = _create_name_or_use_existing_one(container, 'Resize', operator_name)
        attrs = {'name': name}
        attrs['mode'] = mode.lower()
    
        inputs = [input_name]
    
        if container.target_opset < 11:
            op_version = 10
        else:
            op_version = 11
            roi_tensor_name = self.get_unique_tensor_name(name + '_roi')
            roi = [0.0] * len(scales) + [1.0] * len(scales)
            container.add_initializer(roi_tensor_name, onnx_proto.TensorProto.FLOAT, [2 * len(scales)], roi)
            inputs.append(roi_tensor_name)
            attrs['coordinate_transformation_mode'] = coordinate_transformation_mode
            if attrs['mode'] == 'nearest':
                attrs['nearest_mode'] = 'floor'
    
        scales_tensor_name = self.get_unique_tensor_name(name + '_scales')
        container.add_initializer(scales_tensor_name, onnx_proto.TensorProto.FLOAT, [len(scales)], scales)
        inputs.append(scales_tensor_name)
        container.add_node('Resize', inputs, output_name, op_version=op_version, **attrs)
        return output_name

    def rnn(self, input_names, output_names, container, operator_name=None, output_seq=0, **attrs):
        name = _create_name_or_use_existing_one(container, 'RNN', operator_name)
        if container.target_opset <= 6:
            attrs['output_sequence'] = 1 if output_seq else 0
            op_version = 1
        else:
            op_version = 7
        container.add_node('RNN', input_names, output_names, name=name, op_version=op_version, **attrs)
        return output_names

    def shape(self, input_name, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Shape', operator_name)
        container.add_node('Shape', input_name, output_name, name=name, op_version=1)
        return output_name

    def sigmoid(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Sigmoid', input_name, output_name, container, operator_name)
        return output_name

    def softsign(self, input_name, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Softsign', operator_name)
        container.add_node('Softsign', input_name, output_name, name=name, op_version=1)
        return output_name

    # See alpha and gamma at https://github.com/keras-team/keras/blob/master/keras/activations.py#L80-L81
    def selu(self, input_name, output_name, container, operator_name=None, alpha=1.673263, gamma=1.050701):
        self._apply_unary_operation('Selu', input_name, output_name, container, operator_name, alpha=alpha, gamma=gamma)
        return output_name

    def softmax(self, input_name, output_name, container, operator_name=None, axis=None):
        name = _create_name_or_use_existing_one(container, 'Softmax', operator_name)
        if axis is None:
            axis = 1 if container.target_opset < 13 else -1
        container.add_node('Softmax', input_name, output_name, name=name, axis=axis)
        return output_name

    def scaled_tanh(self, input_name, output_name, container, operator_name=None, alpha=None, beta=None):
        if alpha is None:
            alpha = [1.0]
        if beta is None:
            beta = [1.0]
        if len(alpha) != 1 or len(beta) != 1:
            raise ValueError('alpha and beta must be 1-element lists')
    
        name = _create_name_or_use_existing_one(container, 'ScaledTanh', operator_name)
        if container.target_opset < 9:
            attrs = {'name': name, 'alpha': alpha[0], 'beta': beta[0]}
            container.add_node('ScaledTanh', input_name, output_name, **attrs)
        else:
            # Define scalar a, initialize with parameter alpha.
            aName = self.get_unique_tensor_name(name + '_alpha')
            aShape = [len(alpha)] if len(alpha) == 1 else [len(alpha), 1, 1]
            container.add_initializer(aName, onnx_proto.TensorProto.FLOAT, aShape, alpha)
    
            # Define scalar b, initialize with parameter beta.
            bShape = [len(beta)] if len(beta) == 1 else [len(beta), 1, 1]
            bName = self.get_unique_tensor_name(name + '_beta')
            container.add_initializer(bName, onnx_proto.TensorProto.FLOAT, bShape, beta)
    
            # c = b * x
            cName = self.get_unique_tensor_name(name + '_c')
            self.mul([input_name, bName], cName, container)
    
            # d = tanh(c)
            dName = self.get_unique_tensor_name(name + '_d')
            self.tanh(cName, dName, container)
    
            # output = a * d
            self.mul([aName, dName], output_name, container)
        return output_name

    def slice(self, input_name, output_name, container,
              operator_name=None, starts=None, ends=None, axes=None, steps=None):
        assert starts is not None, 'the starts in slice op cannot be None'
        assert ends is not None, 'the ends in slice op cannot be None'
        name = _create_name_or_use_existing_one(container, 'Slice', operator_name)
    
        if container.target_opset < 10:
            if axes is None:
                container.add_node('Slice', input_name, output_name, name=name,
                                   starts=starts, ends=ends, op_version=1)
            else:
                container.add_node('Slice', input_name, output_name, name=name,
                                   starts=starts, ends=ends, axes=axes, op_version=1)
        else:
            if container.target_opset == 10:
                op_version = 10
            else:
                op_version = 11
            inputs = input_name if isinstance(input_name, list) else [input_name]
            if isinstance(starts, str):
                starts_name = starts
            else:
                starts_name = self.get_unique_tensor_name('starts')
                container.add_initializer(starts_name, onnx_proto.TensorProto.INT64,
                                          [len(starts)], starts)
    
            if isinstance(ends, str):
                ends_name = ends
            else:
                ends_name = self.get_unique_tensor_name('ends')
                container.add_initializer(ends_name, onnx_proto.TensorProto.INT64,
                                          [len(ends)], ends)
    
            inputs.append(starts_name)
            inputs.append(ends_name)
            if axes:
                if isinstance(axes, str):
                    axes_name = axes
                else:
                    axes_name = self.get_unique_tensor_name('axes')
                    container.add_initializer(axes_name, onnx_proto.TensorProto.INT64,
                                              [len(axes)], axes)
                inputs.append(axes_name)
            if steps:
                if not axes:
                    inputs.append('')
                if isinstance(steps, str):
                    steps_name = steps
                else:
                    steps_name = self.get_unique_tensor_name('steps')
                    container.add_initializer(steps_name, onnx_proto.TensorProto.INT64,
                                              [len(steps)], steps)
                inputs.append(steps_name)
            container.add_node('Slice', inputs, output_name, name=name,
                               op_version=op_version)
        return output_name
    
    def split(self, input_name, output_names, container, operator_name=None, split=None, axis=0):
        name = _create_name_or_use_existing_one(container, 'Split', operator_name)
        if container.target_opset <= 1:
            op_version = 1
        elif container.target_opset < 11:
            op_version = 2
        elif container.target_opset < 13:
            op_version = 11
        else:
            op_version = 13
    
        attrs = {'name': name}
        if split is not None:
            if container.target_opset < 13:
                attrs['split'] = split
            else:
                if not isinstance(input_name, list):
                    input_name = [input_name]
                if isinstance(split, str):
                    split_name = split
                else:
                    split_name = self.get_unique_tensor_name(name + '_split')
                    container.add_initializer(split_name, onnx_proto.TensorProto.INT64, [len(split)], split)
                input_name = input_name + [split_name]
    
        if axis is not None:
            attrs['axis'] = axis
    
        container.add_node('Split', input_name, output_names, op_version=op_version, **attrs)
        return output_names

    def sqrt(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Sqrt', input_name, output_name, container, operator_name=operator_name)
        return output_name

    def _apply_squeeze_unsqueeze(self, input_name, output_name, container, squeeze_str, operator_name=None, axes=None,
                                 rank=0):
        name = _create_name_or_use_existing_one(container, squeeze_str, operator_name)
        if container.target_opset < 13:
            if container.target_opset < 11:
                op_version = 1
                axes = [axis if axis >= 0 else axis + rank for axis in axes]
            else:
                op_version = 11
            container.add_node(squeeze_str, input_name, output_name, name=name, op_version=op_version, axes=axes)
        else:
            op_version = 13
            if not isinstance(input_name, list):
                input_name = [input_name]
            if isinstance(axes, str):
                container.add_node(squeeze_str, input_name + [axes], output_name, op_version=op_version, name=name)
            elif len(axes) == 0:
                container.add_node(squeeze_str, input_name, output_name, op_version=op_version, name=name)
            else:
                axes_name = self.get_unique_tensor_name(name + '_axes')
                container.add_initializer(axes_name, onnx_proto.TensorProto.INT64, [len(axes)], axes)
                container.add_node(squeeze_str, input_name + [axes_name], output_name, op_version=op_version, name=name)
        return output_name
    
    def squeeze(self, input_name, output_name, container, operator_name=None, axes=None, rank=0):
        if axes is None:
            axes = []
        self._apply_squeeze_unsqueeze(input_name, output_name, container, 'Squeeze', operator_name, axes, rank)
        return output_name
    
    def sub(self, input_names, output_name, container, operator_name=None, axis=None, broadcast=0):
        self._apply_basic_numerical_operation('Sub', input_names, output_name, container, operator_name=operator_name,
                                              axis=axis, broadcast=broadcast)
        return output_name

    def sum(self, input_names, output_name, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'Sum', operator_name)
        if container.target_opset < 6:
            op_version = 1
        else:
            op_version = 6
        container.add_node('Sum', input_names, output_name, op_version=op_version, name=name)
        return output_name
    
    def tanh(self, input_name, output_name, container, operator_name=None):
        self._apply_unary_operation('Tanh', input_name, output_name, container, operator_name)
        return output_name
    
    def thresholded_relu(self, input_name, output_name, container, operator_name=None, alpha=None):
        if alpha is None:
            alpha = [1.0]
    
        name = _create_name_or_use_existing_one(container, 'ThresholdedRelu', operator_name)
        attrs = {'name': name, 'alpha': alpha[0]}
        if container.target_opset < 10:
            # ThresholdedRelu graduated from an experimental op to a full op in opset 10
            # onnxruntime maintains support in the ONNX domain for ThresholdedRelu as a contrib op
            attrs['op_domain'] = "ai.onnx"
            op_version = 1
        else:
            op_version = 10
        container.add_node('ThresholdedRelu', input_name, output_name, op_version=op_version, **attrs)
        return output_name
    
    def tile(self, input_name, output_name, container, operator_name=None, repeats=None):
        name = _create_name_or_use_existing_one(container, 'Tile', operator_name)
    
        if repeats is None or (not isinstance(repeats, str) and all(repeat_count == 1 for repeat_count in repeats)):
            container.add_node('Identity', input_name, output_name, name=name)
            return output_name
    
        if container.target_opset < 6:
            intermediate_input_name = input_name
            intermediate_output_name = None
            if isinstance(repeats, str):
                raise ValueError('repeats cannot be string type before opset 6')
    
            for axis, repeat_count in enumerate(repeats):
                if repeat_count == 1:
                    continue
    
                # Create the 2nd input of Tile
                tile_tensor_name = self.get_unique_tensor_name(name + '_tile')
                container.add_initializer(tile_tensor_name, onnx_proto.TensorProto.FLOAT, [1], [float(repeat_count)])
    
                # Create the 3rd input of Tile
                axis_tensor_name = self.get_unique_tensor_name(name + '_axis')
                container.add_initializer(axis_tensor_name, onnx_proto.TensorProto.FLOAT, [1], [float(axis)])
    
                # Create tile for duplicating along one axis. After ONNX-1.2, we can duplicate along multiple axes,
                # so we don't have to iterate through all axes.
                intermediate_output_name = self.get_unique_tensor_name(name + '_input')
                container.add_node('Tile', [intermediate_input_name, tile_tensor_name, axis_tensor_name],
                                   intermediate_output_name, name=name)
    
                # Use the output produced by this round as the input in the next iteration
                intermediate_input_name = intermediate_output_name
    
                # Create a new name for next Tile
                name = container.get_unique_operator_name('Tile')
    
            # Use the last Tile name for the name of an Identity
            container.add_node('Identity', intermediate_output_name, output_name, op_version=1, name=name)
        else:
            # ONNX-1.2 has a new Tile and we use it here
            if isinstance(repeats, str):
                container.add_node('Tile', input_name + [repeats], output_name, op_version=6, name=name)
            else:
                repeat_tensor_name = self.get_unique_tensor_name(name + '_repeats')
                container.add_initializer(repeat_tensor_name, onnx_proto.TensorProto.INT64, [len(repeats)], repeats)
                container.add_node('Tile', [input_name, repeat_tensor_name], output_name, op_version=6, name=name)
        return output_name
    
    def topk(self, input_name, output_names, container, k, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'TopK', operator_name)
    
        if container.target_opset < 10:
            if isinstance(k, str):
                raise ValueError('topk k cannot be string type before opset 10')
            container.add_node('TopK', input_name, output_names, name=name, k=k, op_version=1)
        else:
            if container.target_opset == 10:
                op_version = 10
            else:
                op_version = 11
    
            if isinstance(k, str):
                k_value_name = k
            else:
                k_value_name = self.get_unique_tensor_name('k_value')
                container.add_initializer(k_value_name, onnx_proto.TensorProto.INT64, [1], [k])
            container.add_node('TopK', input_name + [k_value_name], output_names, name=name, op_version=op_version)
        return output_names
    
    def transpose(self, input_name, output_name, container, operator_name=None, perm=None):
        name = _create_name_or_use_existing_one(container, 'Transpose', operator_name)
        container.add_node('Transpose', input_name, output_name, name=name, perm=perm)
        return output_name
    
    def upsample(self, input_name, output_name, container, operator_name=None, mode='nearest',
                 coordinate_transformation_mode='asymmetric', scales=None):
        """
        :param input_name:
        :param output_name:
        :param container:
        :param operator_name:
        :param mode: nearest or linear
        :param coordinate_transformation_mode:
        :param scales: an integer list of scaling-up rate of all input dimensions
        :return:
        """
        if container.target_opset < 10:
            name = _create_name_or_use_existing_one(container, 'Upsample', operator_name)
            inputs = [input_name]
            attrs = {'name': name}
            if container.target_opset < 7:
                if len(scales) != 4:
                    raise ValueError('Need to specify a 4-element list the the scales of N-, C-, H-, and W-axes')
                attrs['height_scale'] = float(scales[2])
                attrs['width_scale'] = float(scales[3])
                attrs['mode'] = mode.upper()
                op_version = 1
            else:
                attrs['mode'] = mode.lower()
                if container.target_opset < 9:
                    attrs['scales'] = list(map(float, scales))
                    op_version = 7
                else:
                    # scales moved from attribute to input in opset 9
                    scales_tensor_name = self.get_unique_tensor_name(name + '_scales')
                    container.add_initializer(scales_tensor_name, onnx_proto.TensorProto.FLOAT, [len(scales)], scales)
                    inputs = [input_name, scales_tensor_name]
                    op_version = 9
    
            container.add_node('Upsample', inputs, output_name, op_version=op_version, **attrs)
        else:
            # Upsample op is deprecated in ONNX opset 10
            # We implement Upsample through Resize instead
            self.resize(input_name, output_name, container, operator_name, mode, coordinate_transformation_mode,
                         scales)
        return output_name
    
    def unsqueeze(self, input_name, output_name, container, operator_name=None, axes=None, rank=0):
        if axes is None:
            axes = [0]
        self._apply_squeeze_unsqueeze(input_name, output_name, container, 'Unsqueeze', operator_name, axes, rank)
        return output_name

    def where(self, input_names, output_names, container, operator_name=None):
        name = _create_name_or_use_existing_one(container, 'where', operator_name)
        container.add_node('Where', input_names, output_names, op_version=9, name=name)
        return output_names

    def loop(self, input_names, output_names, container, operator_name=None, body=None):
        name = _create_name_or_use_existing_one(container, 'loop', operator_name)
        trip_count, cond, *states = tuple(input_names)
        trip_count = '' if trip_count is None else trip_count
        cond_name = '' if cond is None else cond
        container.add_node(
            'Loop', [trip_count, cond_name] + states, output_names, op_version=11, name=name, body=body)
        return output_names

    def model_call(self, input_name, output_name, container, operator_name=None, oxml=None):
        name = operator_name
        if name is None:
            name = container.get_unique_operator_name('og')

        # The tensor name replacement happens on unfolding ONNX model.
        for idx, nm_ in enumerate(input_name):
            nvi = oxml.graph.input[idx]
            self.identity([nm_], ["{}_{}".format(name, nvi.name)], container)
            container.value_info.append(nvi)
        for idx, nm_ in enumerate(output_name):
            self.identity(["{}_{}".format(name, oxml.graph.output[idx].name)], [nm_], container)
        container.value_info.extend(oxml.graph.output)
        container.add_model_node(input_name, output_name, name=name, model=oxml)
        return output_name


class _ONNXModelBuilder(_ONNXOperatorAPI):
    def __init__(self):
        _OpSchema._ox = self
        self._id_count = 0
        self.opdict_counter = {}

    def get_unique_tensor_name(self, hint):
        self._id_count += 1
        return "v{}_{}".format(hint, str(self._id_count))

    def make_tensor(self, dtype, dims, vals):
        return helper.make_tensor(self.get_unique_tensor_name('ts'), dtype, dims, vals)

    def get_unique_operator_type_name(self, op_type):
        nn = self.opdict_counter.get(op_type, 0)
        self.opdict_counter[op_type] = nn + 1
        return "_Op{}".format(op_type) if nn == 0 else "_Op{}_{}".format(op_type, nn+1)

    @classmethod
    def is_raw(cls, func):  # without any schema decorator
        return not isinstance(func, _OpSchema)


# Singleton
ox = _ONNXModelBuilder()
