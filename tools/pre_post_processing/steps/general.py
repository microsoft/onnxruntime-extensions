# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from typing import List
from ..step import Step


class ReverseAxis(Step):
    """
    Reverses the data in an axis by splitting and concatenating in reverse order.
      e.g. convert RGB ordered data to BGR.
    Output data type and shape is the same as the input.
    """

    def __init__(self, axis: int = -1, dim_value: int = -1, name: str = None):
        """
        Initialize ReverseAxis step.
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

    def _create_graph_for_step(self, graph: onnx.GraphProto):
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

        reverse_graph = onnx.parser.parse_graph(
            f"""\
            reverse_axis ({input_type_str}[{input_shape_str}] {self.input_names[0]})
                => ({input_type_str}[{input_shape_str}] {self.output_names[0]})  
            {{
                {','.join(split_outs)} = Split <axis = {self._axis}> ({self.input_names[0]})
                {self.output_names[0]} = Concat <axis = {self._axis}> ({','.join(reversed(split_outs))})
            }}
            """
        )

        onnx.checker.check_graph(reverse_graph)
        return reverse_graph


class Squeeze(Step):
    """
    ONNX Squeeze
    """

    def __init__(self, axes: List[int] = None, name: str = None):
        super().__init__(["data"], ["squeezed"], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(",")
        output_dims = [dim for idx, dim in enumerate(dims) if idx not in self._axes]
        output_shape_str = ",".join(output_dims)

        if self._axes:
            axes_strs = [str(axis) for axis in self._axes]
            graph_str = f"""\
            axes = Constant <value = int64[{len(self._axes)}] {{{','.join(axes_strs)}}}> ()
            {self.output_names[0]} = Squeeze({self.input_names[0]}, axes)
            """
        else:
            graph_str = f"{self.output_names[0]} = Squeeze({self.input_names[0]})"

        squeeze_graph = onnx.parser.parse_graph(
            f"""\
            squeeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                {graph_str}
            }}
            """
        )

        return squeeze_graph


class Transpose(Step):
    """
    ONNX Transpose.
    """

    def __init__(self, perms: List[int], name: str = None):
        super().__init__(["X"], ["transposed"], name)
        self.perms = perms

    def _create_graph_for_step(self, graph: onnx.GraphProto):
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

    def __init__(self, name: str = None):
        super().__init__(["data"], ["probabilities"], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
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

    def __init__(self, axes: List[int], name: str = None):
        super().__init__(["data"], ["expanded"], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto):
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
