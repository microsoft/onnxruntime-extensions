# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import onnx

from onnx import parser
from typing import List, Optional, Tuple

from .utils import (
    IoMapEntry,
    create_custom_op_checker_context,
    TENSOR_TYPE_TO_ONNX_TYPE,
)


class Step(object):
    """Base class for a pre or post processing step."""

    prefix = "_ppp"
    _step_num = 0  # unique step number so we can prefix the naming in the graph created for the step
    _custom_op_checker_context = create_custom_op_checker_context()

    def __init__(self, inputs: List[str], outputs: List[str], name: Optional[str] = None):
        """
        Initialize the step.

        Args:
            inputs: List of default input names.
            outputs: List of default output names.
            name: Step name. Defaults to the derived class name.
        """
        self.step_num = Step._step_num
        self.input_names = inputs
        self.output_names = outputs
        self.name = name if name else f"{self.__class__.__name__}"
        self._prefix = f"{Step.prefix}{self.step_num}_"

        Step._step_num += 1

    def connect(self, entry: IoMapEntry):
        """
        Connect the value name from a previous step to an input of this step so they match.
        This makes joining the GraphProto created by each step trivial.
        """
        assert len(entry.producer.output_names) >= entry.producer_idx
        assert len(self.input_names) >= entry.consumer_idx
        assert isinstance(entry.producer, Step)

        self.input_names[entry.consumer_idx] = entry.producer.output_names[entry.producer_idx]

    def apply(self, graph: onnx.GraphProto):
        """Append the nodes that implement this step to the provided graph."""

        graph_for_step = self._create_graph_for_step(graph)
        onnx.checker.check_graph(graph_for_step, Step._custom_op_checker_context)

        # prefix the graph for this step to guarantee no clashes of value names with the existing graph
        onnx.compose.add_prefix_graph(graph_for_step, self._prefix, inplace=True)
        result = self.__merge(graph, graph_for_step)

        # update self.output_names to the prefixed names so that when we connect later Steps the values match
        new_outputs = [self._prefix + o for o in self.output_names]
        result_outputs = [o.name for o in result.output]

        # sanity check that all of our outputs are in the merged graph
        for o in new_outputs:
            assert o in result_outputs

        self.output_names = new_outputs

        return result

    @abc.abstractmethod
    def _create_graph_for_step(self, graph: onnx.GraphProto):
        """Derived class should implement this and return the GraphProto containing the nodes required to
        implement the step."""
        pass

    def __merge(self, first: onnx.GraphProto, second: onnx.GraphProto):
        # We prefixed all the value names in `second`, so allow for that when connecting the two graphs
        io_map = []
        for o in first.output:
            # apply the same prefix to the output from the previous step to match the prefixed graph from this step
            prefixed_output = self._prefix + o.name
            for i in second.input:
                if i.name == prefixed_output:
                    io_map.append((o.name, i.name))

        outputs_to_preserve = None

        # special handling of Debug class.
        if isinstance(self, Debug):
            # preserve outputs of the first graph so they're available downstream. otherwise they are consumed by
            # the Debug node and disappear during the ONNX graph_merge as it considers consumed values to be
            # internal - which is entirely reasonable when merging graphs.
            # the issue we have is that we don't know what future steps might want things to remain as outputs.
            # the current approach is to insert a Debug step which simply duplicates the values so that they are
            # guaranteed not be consumed (only one of the two copies will be used).
            # doesn't change the number of outputs from the previous step, so it can be transparently inserted in the
            # pre/post processing pipeline.
            # need to also list the second graph's outputs when manually specifying outputs.
            outputs_to_preserve = [o.name for o in first.output] + [o.name for o in second.output]

        # merge with existing graph
        merged_graph = onnx.compose.merge_graphs(first, second, io_map, outputs=outputs_to_preserve)

        return merged_graph

    @staticmethod
    def _elem_type_str(elem_type: int):
        return TENSOR_TYPE_TO_ONNX_TYPE[elem_type]

    @staticmethod
    def _shape_to_str(shape: onnx.TensorShapeProto):
        """Returns the values from the shape as a comma separated string."""

        def dim_to_str(dim):
            if dim.HasField("dim_value"):
                return str(dim.dim_value)
            elif dim.HasField("dim_param"):
                return dim.dim_param
            else:
                return ""

        shape_str = ",".join([dim_to_str(dim) for dim in shape.dim])
        return shape_str

    def _input_tensor_type(self, graph: onnx.GraphProto, input_num: int) -> onnx.TensorProto:
        """Get the onnx.TensorProto for the input from the outputs of the graph we're appending to."""

        input_type = None
        for o in graph.output:
            if o.name == self.input_names[input_num]:
                input_type = o.type.tensor_type
                break

        if not input_type:
            raise ValueError(f"Input {self.input_names[input_num]} was not found in outputs of graph.")

        return input_type

    def _get_input_type_and_shape_strs(self, graph: onnx.GraphProto, input_num: int) -> Tuple[str, str]:
        input_type = self._input_tensor_type(graph, input_num)
        return Step._elem_type_str(input_type.elem_type), Step._shape_to_str(input_type.shape)


# special case. we include the helper Debug step here as logic in the base class is conditional on it.
class Debug(Step):
    """
    Step that can be arbitrarily inserted in the pre or post processing pipeline.
    It will make the outputs of the previous Step also become graph outputs so their value can be more easily debugged.

    NOTE: Depending on when the previous Step's outputs are consumed in the pipeline the graph output for it
          may or may not have '_debug' as a suffix.
          TODO: PrePostProcessor __cleanup_graph_output_names could also hide the _debug by inserting an Identity node
                to rename so it's more consistent.
    """

    def __init__(self, num_inputs: int = 1, name: Optional[str] = None):
        """
        Initialize Debug step
        Args:
            num_inputs: Number of inputs from previous Step to make graph outputs.
            name: Optional name for Step. Defaults to 'Debug'
        """
        self._num_inputs = num_inputs
        input_names = [f"input{i}" for i in range(0, num_inputs)]
        output_names = [f"debug{i}" for i in range(0, num_inputs)]

        super().__init__(input_names, output_names, name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_str = ""
        output_str = ""
        output_debug_str = ""
        nodes_str = ""

        # update output names so we preserve info from the latest input names
        self.output_names = [f"{name}_debug" for name in self.input_names]

        for i in range(0, self._num_inputs):
            input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, i)
            if i > 0:
                input_str += ", "
                output_str += ", "
                output_debug_str += ", "
                nodes_str += "\n"

            input_str += f"{input_type_str}[{input_shape_str}] {self.input_names[i]}"
            output_str += f"{input_type_str}[{input_shape_str}] {self.output_names[i]}"
            nodes_str += f"{self.output_names[i]} = Identity({self.input_names[i]})\n"

        debug_graph = onnx.parser.parse_graph(
            f"""\
            debug ({input_str}) 
                => ({output_str})
            {{
                {nodes_str}
            }}
            """
        )

        onnx.checker.check_graph(debug_graph)
        return debug_graph
