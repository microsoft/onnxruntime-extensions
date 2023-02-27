# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from typing import List, Optional
from ..step import Step


class ReverseAxis(Step):
    """
    Reverses the data in an axis by splitting and concatenating in reverse order.
      e.g. convert RGB ordered data to BGR.
    Output data type and shape is the same as the input.
    """

    def __init__(self, axis: int = -1, dim_value: int = -1, name: Optional[str] = None):
        """
        Args:
            axis: Axis to reverse. Default is last axis.
            dim_value: Explicit value for size of dimension being reversed.
                       This can be provided if the axis being reversed currently has a symbolic value.
                       Note that this will fail during graph execution if the actual value at runtime does not match.
                       If not provided, the size of the dimension to reverse is inferred from the input shape.
            name: Optional Step name. Defaults to 'ReverseAxis'
        """
        super().__init__(["data"], ["data_with_reversed_axis"], name)
        self._axis = axis
        self._dim_value = dim_value

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        input_dims = input_shape_str.split(",")
        split_dim = input_dims[self._axis]

        if split_dim.isdigit():
            dim_value = int(split_dim)
            if self._dim_value != -1:
                # TODO: Technically we don't require a match here. For now expect it to match.
                assert dim_value == self._dim_value
            else:
                self._dim_value = dim_value

        split_outs = []
        for i in range(0, self._dim_value):
            split_outs.append(f"split_out_{i}")

        split_attr = f"axis = {self._axis}"
        if onnx_opset >= 18:
            # Split now requires the number of outputs to be specified even though that can be easily inferred...
            split_attr += f", num_outputs = {len(split_outs)}"

        reverse_graph = onnx.parser.parse_graph(
            f"""\
            reverse_axis ({input_type_str}[{input_shape_str}] {self.input_names[0]})
                => ({input_type_str}[{input_shape_str}] {self.output_names[0]})  
            {{
                {','.join(split_outs)} = Split <{split_attr}> ({self.input_names[0]})
                {self.output_names[0]} = Concat <axis = {self._axis}> ({','.join(reversed(split_outs))})
            }}
            """
        )

        return reverse_graph


class Squeeze(Step):
    """
    ONNX Squeeze
    """

    def __init__(self, axes: Optional[List[int]] = None, name: Optional[str] = None):
        """
        Args:
            axes: Axes to remove.
                  If None, remove all axes with size of 1. Requires all dimensions to have explicit values.
            name: Optional Step name. Defaults to 'Squeeze'
        """
        super().__init__(["data"], ["squeezed"], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(",")

        axes = self._axes
        if not axes:
            axes = []
            for idx, dim in enumerate(dims):
                if not dim.isnumeric():
                    # we can't infer the output shape if there are symbolic dims
                    raise ValueError("Axes must be specified if there are symbolic dimensions.")

                if dim == '1':
                    axes.append(int(idx))

        output_dims = [dim for idx, dim in enumerate(dims) if idx not in axes]
        output_shape_str = ",".join(output_dims)

        axes_strs = [str(axis) for axis in axes]

        squeeze_graph = onnx.parser.parse_graph(
            f"""\
            squeeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                axes = Constant <value = int64[{len(axes)}] {{{','.join(axes_strs)}}}> ()
                {self.output_names[0]} = Squeeze({self.input_names[0]}, axes)
            }}
            """
        )

        return squeeze_graph


class Transpose(Step):
    """
    ONNX Transpose.
    """

    def __init__(self, perms: List[int], name: Optional[str] = None):
        """
        Args:
            perms: List of integers with permutations to apply.
            name: Optional Step name. Defaults to 'Transpose'
        """
        super().__init__(["X"], ["transposed"], name)
        self.perms = perms

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        perms_str = ",".join([str(idx) for idx in self.perms])
        dims = input_shape_str.split(",")
        output_dims = [dims[axis] for axis in self.perms]
        output_shape_str = ",".join(output_dims)

        transpose_graph = onnx.parser.parse_graph(
            f"""\
            transpose ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = Transpose <perm = [{perms_str}]> ({self.input_names[0]})
            }}
            """
        )

        return transpose_graph


class Softmax(Step):
    """
    ONNX Softmax
    """

    def __init__(self, name: Optional[str] = None):
        """
        Args:
            name: Optional Step name. Defaults to 'Softmax'
        """
        super().__init__(["data"], ["probabilities"], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)

        softmax_graph = onnx.parser.parse_graph(
            f"""\
            softmax ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{input_shape_str}] {self.output_names[0]})
            {{
                {self.output_names[0]} = Softmax ({self.input_names[0]})
            }}
            """
        )

        return softmax_graph


class Unsqueeze(Step):
    """
    ONNX Unsqueeze
    """

    def __init__(self, axes: List[int], name: Optional[str] = None):
        """
        Args:
            axes: List of integers indicating the dimensions to be inserted.
            name: Optional Step name. Defaults to 'Unsqueeze'
        """
        super().__init__(["data"], ["expanded"], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(",")

        for idx in self._axes:
            dims.insert(idx, "1")

        output_shape_str = ",".join(dims)
        axes_strs = [str(axis) for axis in self._axes]

        unsqueeze_graph = onnx.parser.parse_graph(
            f"""\
            unsqueeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                axes = Constant <value = int64[{len(self._axes)}] {{{','.join(axes_strs)}}}> ()
                {self.output_names[0]} = Unsqueeze ({self.input_names[0]}, axes)
            }}
            """
        )

        return unsqueeze_graph


class ArgMax(Step):
    def __init__(self, name: Optional[str] = None, axis: int = -1, keepdims: int = 0):
        """
        Brief:
            Same as ArgMax op.
        Args:
            name: Optional name of step. Defaults to 'ArgMax'

        """
        super().__init__(["data"], ["index"], name)
        self._axis = axis
        self._keepdims = keepdims

    def _create_graph_for_step(self, graph: onnx.GraphProto, onnx_opset: int):
        input_type_str_0, input_shape_str_0 = self._get_input_type_and_shape_strs(graph, 0)
        input_shape_0 = input_shape_str_0.split(",")

        def build_input_declare():
            return f"{input_type_str_0}[{input_shape_str_0}] {self.input_names[0]}"

        axis = self._axis + len(input_shape_0) if self._axis < 0 else self._axis
        if axis >= len(input_shape_0):
            raise ValueError("axis should be in range [-rank, rank-1].")
        
        output_shape_str = input_shape_0.copy()
        output_shape_str[axis] = "1"
        if self._keepdims == 0:
            output_shape_str.pop(axis)

        converter_graph = onnx.parser.parse_graph(
            f"""\
            classify ({build_input_declare()}) 
                => (int64[{','.join(output_shape_str)}] {self.output_names[0]})
            {{
                {self.output_names[0]} = ArgMax<axis = {self._axis}, keepdims={self._keepdims}>({self.input_names[0]})
            }}
            """
        )

        return converter_graph
