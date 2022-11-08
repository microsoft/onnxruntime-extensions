import abc
import argparse
import enum
import numpy as np
import onnx
import os

from dataclasses import dataclass
from onnx import parser, version_converter
from pathlib import Path
from typing import List, Tuple, Union

# The ONNX graph parser has it's own map of names just to be special
# https://github.com/onnx/onnx/blob/604af9cb28f63a6b9924237dcb91530649233db9/onnx/defs/parser.h#L72
TENSOR_TYPE_TO_ONNX_GRAPH_TYPE = {
    int(onnx.TensorProto.FLOAT): 'float',
    int(onnx.TensorProto.UINT8): 'uint8',
    int(onnx.TensorProto.INT8): 'int8',
    int(onnx.TensorProto.UINT16): 'uint16',
    int(onnx.TensorProto.INT16): 'int16',
    int(onnx.TensorProto.INT32): 'int32',
    int(onnx.TensorProto.INT64): 'int64',
    int(onnx.TensorProto.STRING): 'string',
    int(onnx.TensorProto.BOOL): 'bool',
    int(onnx.TensorProto.FLOAT16): 'float16',
    int(onnx.TensorProto.DOUBLE): 'double',
    int(onnx.TensorProto.UINT32): 'uint32',
    int(onnx.TensorProto.UINT64): 'uint64',
    int(onnx.TensorProto.COMPLEX64): 'complex64',
    int(onnx.TensorProto.COMPLEX128): 'complex128',
    int(onnx.TensorProto.BFLOAT16): 'bfloat16',
}

# We need to use an opset that's valid for the pre/post processing operators we add.
# Could alternatively use onnx.defs.onnx_opset_version to match the onnx version installed, but that's not deterministic
# For now it's an arbitrary default of ONNX v16.
PRE_POST_PROCESSING_ONNX_OPSET = 16


def get_opset_imports():
    return {'': PRE_POST_PROCESSING_ONNX_OPSET,
            'com.microsoft.ext': 1,
            'ai.onnx.contrib': 1}


# Create an checker context that includes the ort-ext domain so that custom ops don't cause failure
def _create_custom_op_checker_context():
    context = onnx.checker.C.CheckerContext()
    context.ir_version = onnx.checker.DEFAULT_CONTEXT.ir_version
    context.opset_imports = get_opset_imports()

    return context


@dataclass
class IoMapEntry:
    """Entry to map the output index from a producer to the input index of a consumer."""
    # optional producer value
    #   producer is inferred from previous step if not provided.
    #   producer Step will be search for by PrePostProcessor using name if str is provided
    producer: Union["Step", str] = None
    producer_idx: int = 0
    consumer_idx: int = 0


class Step(object):
    """Base class for a pre or post processing step"""

    prefix = '_ppp'
    _step_num = 0  # unique step number so we can prefix the naming in the graph created for the step
    _custom_op_checker_context = _create_custom_op_checker_context()

    def __init__(self, inputs: List[str], outputs: List[str], name: str = None):
        self.step_num = Step._step_num
        self.input_names = inputs
        self.output_names = outputs
        self.name = name if name else f'{self.__class__.__name__}'
        self._prefix = f'{Step.prefix}{self.step_num}_'

        Step._step_num += 1

    def connect(self, entry: IoMapEntry):
        """
        Connect the value name from a previous step to an input of this step so we use matching value names.
        This makes joining the GraphProto created by each step trivial.
        """
        assert(len(entry.producer.output_names) >= entry.producer_idx)
        assert(len(self.input_names) >= entry.consumer_idx)
        assert(isinstance(entry.producer, Step))
        self.input_names[entry.consumer_idx] = entry.producer.output_names[entry.producer_idx]

    def apply(self, graph: onnx.GraphProto):
        """Append the nodes that implement this step to the provided graph."""

        graph_for_step = self._create_graph_for_step(graph)

        # TODO: The onnx renamer breaks the graph input/output name mapping between steps as rename_inputs and
        # rename_outputs only applies if we're not updating all the other values (rename_edges=True)
        # It's better to rename each graph so we guarantee no clashes though
        onnx.compose.add_prefix_graph(graph_for_step, self._prefix, inplace=True)
        result = self.__merge(graph, graph_for_step)

        # update self.output_names to the prefixed names so that when we connect later Steps the values match
        new_outputs = [self._prefix + o for o in self.output_names]
        result_outputs = [o.name for o in result.output]

        # sanity check that all of our outputs are in the merged graph
        for o in new_outputs:
            assert(o in result_outputs)

        self.output_names = new_outputs

        return result

    @abc.abstractmethod
    def _create_graph_for_step(self, graph: onnx.GraphProto):
        pass

    def __merge(self, first: onnx.GraphProto, second: onnx.GraphProto):
        # We prefixed everything in `second` so allow for that when connection the two graphs
        io_map = []
        for o in first.output:
            # apply the same prefix to the output from the previous step to match the prefixed graph from this step
            prefixed_output = self._prefix + o.name
            for i in second.input:
                if i.name == prefixed_output:
                    io_map.append((o.name, i.name))

        outputs_to_preserve = None

        # special handling of Debug class. TBD if there's a better way to do this.
        if isinstance(self, Debug):
            # preserve outputs of the first graph so they're available downstream. otherwise they are consumed by
            # the Debug node and disappear during the ONNX graph merge as it considered consumed values to become
            # internal - which is entirely reasonable when merging graphs.
            # the issue we have is that we don't know what future steps might want things to remain as outputs.
            # the current approach is to insert a Debug step which simply duplicates the values so they are
            # guaranteed not be consumed. also doesn't change the number of outputs so it can be transparently inserted.
            #
            # need to also list the second graph's outputs when manually specifying outputs
            outputs_to_preserve = [o.name for o in first.output] + [o.name for o in second.output]

        # merge with existing graph
        merged_graph = onnx.compose.merge_graphs(first, second, io_map, outputs=outputs_to_preserve)

        return merged_graph

    @staticmethod
    def _elem_type_str(type: int):
        return TENSOR_TYPE_TO_ONNX_GRAPH_TYPE[type]

    @staticmethod
    def _shape_to_str(shape: onnx.TensorShapeProto):
        def dim_to_str(dim):
            if dim.HasField("dim_value"):
                return str(dim.dim_value)
            elif dim.HasField("dim_param"):
                return dim.dim_param
            else:
                return ""

        shape_str = ','.join([dim_to_str(dim) for dim in shape.dim])
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


class PrePostProcessor:
    """
    Class to handle running all the pre/post processing steps and updating the model.
    """
    def __init__(self, inputs: List[onnx.ValueInfoProto] = None, outputs: List[onnx.ValueInfoProto] = None):
        self.pre_processors = []
        self.post_processors = []

        # Connections for each pre/post processor. 1:1 mapping with entries in pre_processors/post_processors
        self._pre_processor_connections = []  # type: List[List[IoMapEntry]]
        self._post_processor_connections = []  # type: List[List[IoMapEntry]]

        # explicitly join outputs from Steps in pre_processors to inputs of the original model
        # format is Step or step name, step_idx, name of graph input/output
        # Pre-processing we connect Step output to original model:
        #   - step_idx is for Step.output_names, and name is in graph.input
        # Post-processing we connect the original model output to the Step input
        #   - step_idx is for Step.input_names, and name is in graph.output
        self._pre_processing_joins = None  # type: Union[None,List[Tuple[Union[Step,str], int, str]]]
        self._post_processing_joins = None  # type: Union[None,List[Tuple[Union[Step,str], int, str]]]

        self._inputs = inputs if inputs else []
        self._outputs = outputs if outputs else []

    def add_pre_processing(self, items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]]):
        """
        Add the pre-processing steps.
        Options are:
          Add Step with default connection of outputs from the previous step (if available) to inputs of this step.
          Add tuple of Step or step name and io_map for connections between two steps.
            Previous step is inferred if IoMapEntry.producer is not specified.
        """
        self.__add_processing(self.pre_processors, self._pre_processor_connections, items)

    def add_post_processing(self, items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]]):
        self.__add_processing(self.post_processors, self._post_processor_connections, items)

    def add_joins(self,
                  preprocessing_joins: List[Tuple[Step, int, str]] = None,
                  postprocessing_joins: List[Tuple[Step, int, str]] = None):
        if preprocessing_joins:
            for step, step_idx, graph_input in preprocessing_joins:
                assert(step and step_idx <= len(step.output_names))

            self._pre_processing_joins = preprocessing_joins

        if postprocessing_joins:
            for step, step_idx, graph_output in postprocessing_joins:
                assert(step and step_idx <= len(step.input_names))

            self._post_processing_joins = postprocessing_joins

    def _add_connection(self, consumer: Step, entry: IoMapEntry):
        producer = self.__producer_from_step_or_str(entry.producer)

        if not((producer in self.pre_processors or producer in self.post_processors) or
                (consumer in self.pre_processors or consumer in self.post_processors)):
            raise ValueError("Producer and Consumer processors must both be registered")

        if producer in self.pre_processors:
            if (consumer in self.pre_processors and
                    self.pre_processors.index(producer) > self.pre_processors.index(consumer)):
                raise ValueError("Producer was registered after consumer and cannot be connected")
        elif producer in self.post_processors:
            if consumer not in self.post_processors:
                raise ValueError("Cannot connect pre-processor consumer with post-processor producer")
            elif self.post_processors.index(producer) > self.post_processors.index(consumer):
                raise ValueError("Producer was registered after consumer and cannot be connected")

        assert(isinstance(producer, Step))
        consumer.connect(entry)

    def run(self, model: onnx.ModelProto):
        # update to the ONNX opset we're using
        model_opset = [entry.version for entry in model.opset_import
                       if entry.domain == '' or entry.domain == 'ai.onnx'][0]

        if model_opset > PRE_POST_PROCESSING_ONNX_OPSET:
            # It will probably work if the user updates PRE_POST_PROCESSING_ONNX_OPSET to match the model
            # but there are no guarantees.
            # Would only break if ONNX operators used in the pre/post processing graphs have had spec changes.
            raise ValueError(f"Model opset is {model_opset} which is newer than the opset used by this script.")

        model = onnx.version_converter.convert_version(model, PRE_POST_PROCESSING_ONNX_OPSET)

        def name_nodes(new_graph: onnx.GraphProto, prefix: str):
            idx = 0
            for n in new_graph.node:
                if not n.name:
                    n.name = prefix + str(idx)
                    idx += 1

        def connect_and_run(graph: onnx.GraphProto, processor: Step, connections: List[IoMapEntry]):
            for connection in connections:
                assert(connection.producer)
                self._add_connection(processor, connection)

            return processor.apply(graph)

        graph = model.graph
        # add pre-processing
        if self.pre_processors:
            # create empty graph with pass through of the requested input name
            pre_process_graph = onnx.GraphProto()
            for i in self._inputs:
                pre_process_graph.input.append(i)
                pre_process_graph.output.append(i)

            for idx, step in enumerate(self.pre_processors):
                pre_process_graph = connect_and_run(pre_process_graph, step, self._pre_processor_connections[idx])

            # name all the nodes for easier debugging
            name_nodes(pre_process_graph, "pre_process_")

            if not self._pre_processing_joins:
                # default to 1:1 between outputs of last step with inputs of model
                last_step = self.pre_processors[-1]
                num_entries = min(len(last_step.output_names), len(graph.input))
                self._pre_processing_joins = [(last_step, i, graph.input[i].name) for i in range(0, num_entries)]

            # map the pre-processing outputs to graph inputs
            io_map = []  # type: List[Tuple[str, str]]
            for step, step_idx, graph_input in self._pre_processing_joins:
                io_map.append((step.output_names[step_idx], graph_input))

            graph = onnx.compose.merge_graphs(pre_process_graph, graph, io_map)

        # add post-processing
        if self.post_processors:
            orig_model_outputs = [o.name for o in model.graph.output]
            graph_outputs = [o.name for o in graph.output]  # this may have additional outputs from pre-processing

            # create default joins if needed
            if not self._post_processing_joins:
                # default to 1:1 between outputs of original model with inputs of first post-processing step
                first_step = self.post_processors[0]
                num_entries = min(len(first_step.input_names), len(orig_model_outputs))
                self._post_processing_joins = [(first_step, i, orig_model_outputs[i]) for i in range(0, num_entries)]

            # update the input names for the steps to match the values produced by the model
            for step, step_idx, graph_output in self._post_processing_joins:
                assert(graph_output in graph_outputs)
                step.input_names[step_idx] = graph_output

            # create empty graph with the values that will be available to the post-processing
            # we do this so we can create a standalone graph for easier debugging
            post_process_graph = onnx.GraphProto()
            for o in graph.output:
                post_process_graph.input.append(o)
                post_process_graph.output.append(o)

            for idx, step in enumerate(self.post_processors):
                post_process_graph = connect_and_run(post_process_graph, step, self._post_processor_connections[idx])

            name_nodes(post_process_graph, "post_process_")

            # io_map should be 1:1 with the post-processing graph given we updated the step input names to match
            io_map = [(o, o) for o in graph_outputs]
            graph = onnx.compose.merge_graphs(graph, post_process_graph, io_map)

        # Make the output names nicer by removing prefixing from naming when applying steps
        graph = PrePostProcessor.__cleanup_graph_output_names(graph)

        new_model = onnx.helper.make_model(graph)

        for domain, opset in get_opset_imports().items():
            # 'if' condition skips onnx domain which is an empty string
            if domain:
                custom_op_import = new_model.opset_import.add()
                custom_op_import.domain = domain
                custom_op_import.version = opset

        onnx.checker.check_model(new_model)
        return new_model

    def __add_processing(self,
                         processors: List[Step],
                         processor_connections: List[List[IoMapEntry]],
                         items: List[Union[Step, Tuple[Union[Step,str], List[IoMapEntry]]]]):
        """
        Add the pre/post processing steps and join with existing steps.

        Args:
            processors: List of processors to add
            processor_connections: Manual connections to create between the pre/post processing graph and the model
            items: List of processors to add.
                   If a Step instances are provided the step will be joined with the immediately previous step.
                   Explicit IoMapEntries can also be provided to join with arbitrary previous steps. The previous step
                   instance can be provided, or it can be lookup up by name.
        """

        for item in items:
            step = None
            explicit_io_map_entries = None

            if isinstance(item, Step):
                step = item
            elif isinstance(item, tuple):
                step_or_str, explicit_io_map_entries = item
                step = self.__producer_from_step_or_str(step_or_str)
            else:
                raise ValueError("Unexpected type " + str(type(item)))

            # start with implicit joins and replace with explicitly provided ones
            # this allows the user to specify the minimum number of manual joins
            io_map_entries = [None] * len(step.input_names)  # type: List[Union[None,IoMapEntry]]
            prev_step = None if len(processors) == 0 else processors[-1]
            if prev_step:
                # default is connecting as many outputs from the previous step as possible
                for i in range(0, min(len(prev_step.output_names), len(step.input_names))):
                    io_map_entries[i] = IoMapEntry(prev_step, i, i)

            # add explicit connections
            if explicit_io_map_entries:
                for entry in explicit_io_map_entries:
                    if not entry.producer:
                        producer = prev_step
                    else:
                        producer = self.__producer_from_step_or_str(entry.producer)

                    io_map_entries[entry.consumer_idx] = IoMapEntry(producer, entry.producer_idx, entry.consumer_idx)

            processors.append(step)
            processor_connections.append([entry for entry in io_map_entries if entry is not None])

    def __producer_from_step_or_str(self, entry: Union[Step,str]):
        if isinstance(entry, Step):
            return entry
        if isinstance(entry, str):
            # search for existing pre or post processing step by name.
            match = (next((s for s in self.pre_processors if s.name == entry), None) or
                     next((s for s in self.post_processors if s.name == entry), None))

            if not match:
                raise ValueError(f'Step named {entry} was not found')

            return match

    @staticmethod
    def __cleanup_graph_output_names(graph: onnx.GraphProto):
        # for each output create identity node to remove prefixing
        io_map = []
        fixes = onnx.GraphProto()
        fixes.input.extend(graph.output)

        # manually handle naming clashes as we don't want to prefix eveything
        input_names = set([i.name for i in graph.input])
        used_names = set(input_names)
        conflicts = 0

        for o in graph.output:
            io_map.append((o.name, o.name))
            clean_name = o.name
            while clean_name.startswith(Step.prefix):
                # output from last step will have one prefixing stage that adds Step._prefix + '_'
                # e.g. '_ppp8_<orig_name>'
                next_underscore = clean_name.find('_', 1)
                if next_underscore > 0:
                    # this check shouldn't be necessary as we always add the trailing '_' when prefixing...
                    if len(clean_name) > next_underscore + 1:
                        next_underscore += 1
                    clean_name = clean_name[next_underscore:]

            # handle things like super resolution where there's an 'image' input and 'image' output
            if clean_name in input_names:
                clean_name += "_out"

            if clean_name in used_names:
                # duplicate - possible when adding debug outputs multiple times
                conflicts += 1
                clean_name += str(conflicts)

            used_names.add(clean_name)

            renamer = onnx.helper.make_node("Identity", [o.name], [clean_name], f"Rename {o.name}")
            fixes.node.append(renamer)

            new_output = fixes.output.add()
            new_output.name = clean_name
            new_output.type.CopyFrom(o.type)

        fixed_graph = onnx.compose.merge_graphs(graph, fixes, io_map)
        return fixed_graph


#
# Pre/Post processing steps
#
class ConvertImageToBGR(Step):
    def __init__(self, name: str = None):
        super().__init__(['image'], ['bgr_data'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert(input_type_str == 'uint8')
        output_shape_str = f'to_bgr_ppp_{self.step_num}_h, to_bgr_ppp_{self.step_num}_w, 3'

        converter_graph = onnx.parser.parse_graph(f'''\
            image_to_bgr (uint8[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} =  com.microsoft.ext.DecodeImage({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


class ConvertBGRToImage(Step):
    def __init__(self, image_format: str = 'jpg', name: str = None):
        super().__init__(['bgr_data'], ['image'], name)
        assert(image_format == 'jpg' or image_format == 'png')
        self._format = image_format

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert(input_type_str == 'uint8')
        output_shape_str = f'to_image_ppp_{self.step_num}_num_bytes'

        converter_graph = onnx.parser.parse_graph(f'''\
            bgr_to_image (uint8[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = com.microsoft.ext.EncodeImage ({self.input_names[0]})
            }}
            ''')

        # as this is a custom op we have to add the attribute for format directly to the node as parse_graph
        # doesn't know the type to validate it.
        format_attr = converter_graph.node[0].attribute.add()
        format_attr.name = "format"
        format_attr.type = onnx.AttributeProto.AttributeType.STRING
        format_attr.s = bytes(self._format, 'utf-8')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


class Resize(Step):
    """
    Resize.
    Aspect ratio is maintained.
    """
    def __init__(self, resize_to: Union[int, Tuple[int, int]], layout: str = "HWC", name: str = None):
        super().__init__(['image'], ['resized_image'], name)
        if isinstance(resize_to, int):
            self._height = self._width = resize_to
        else:
            assert(isinstance(resize_to, tuple))
            self._height, self._width = resize_to

        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')

        # adjust for layout
        # resize will use the largest ratio so both sides won't necessary match the requested height and width.
        # use symbolic names for the output dims as we have to provide values. prefix the names to try and
        # avoid any clashes
        scales_constant_str = 'f_1 = Constant <value = float[1] {1.0}> ()'
        if self._layout == 'HWC':
            assert(len(dims) == 3)
            split_str = "h, w, c"
            scales_str = "ratio_resize, ratio_resize, f_1"
            output_shape_str = f'resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w, {dims[-1]}'
        elif self._layout == 'CHW':
            assert(len(dims) == 3)
            split_str = "c, h, w"
            scales_str = "f_1, ratio_resize, ratio_resize"
            output_shape_str = f'{dims[0]}, resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w'
        elif self._layout == 'HW':
            assert(len(dims) == 2)
            split_str = 'h, w'
            scales_str = "ratio_resize, ratio_resize"
            scales_constant_str = ''
            output_shape_str = f'resize_ppp_{self.step_num}_h, resize_ppp_{self.step_num}_w'
        else:
            raise ValueError(f'Unsupported layout of {self._layout}')

        # TODO: Make this configurable. Matching PIL resize for now
        resize_attributes = "mode = \"linear\", nearest_mode = \"floor\""

        resize_graph = onnx.parser.parse_graph(f'''\
            resize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) => 
                ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_size = Constant <value=float[2] {{{float(self._height)}, {float(self._width)}}}> ()
                image_shape = Shape ({self.input_names[0]})
                {split_str} = Split <axis=0> (image_shape)
                hw = Concat <axis = 0> (h, w)
                f_hw = Cast <to=1> (hw)
                ratios = Div (target_size, f_hw)
                ratio_resize = ReduceMax (ratios)

                {scales_constant_str}
                scales_resize = Concat <axis = 0> ({scales_str})
                {self.output_names[0]} = Resize <{resize_attributes}> ({self.input_names[0]}, , scales_resize)
            }}
            ''')

        onnx.checker.check_graph(resize_graph)
        return resize_graph


class CenterCrop(Step):
    def __init__(self, height: int, width: int, name: str = None):
        super().__init__(['image'], ['cropped_image'], name)
        self._height = height
        self._width = width

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')
        output_shape_str = f'{self._height}, {self._width}, {dims[-1]}'

        crop_graph = onnx.parser.parse_graph(f'''\
            crop ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})
            {{
                target_crop = Constant <value = int64[2] {{{self._height}, {self._width}}}> ()
                i64_2 = Constant <value = int64[1] {{2}}> ()
                axes = Constant <value = int64[2] {{0, 1}}> ()
                x_shape = Shape ({self.input_names[0]})
                h, w, c = Split <axis = 0> (x_shape)
                hw = Concat <axis = 0> (h, w)
                hw_diff = Sub (hw, target_crop)
                start_xy = Div (hw_diff, i64_2)
                end_xy = Add (start_xy, target_crop)
                {self.output_names[0]} = Slice ({self.input_names[0]}, start_xy, end_xy, axes)
            }}
            ''')

        onnx.checker.check_graph(crop_graph)
        return crop_graph


class ImageBytesToFloat(Step):
    """
    Convert uint8 or float pixel values in range 0..255 to floating point in range 0..1
    """
    def __init__(self, name: str = None):
        super().__init__(['data'], ['float_data'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        if input_type_str == 'uint8':
            optional_cast = f'''\
                input_f = Cast <to=1> ({self.input_names[0]})
            '''
        else:
            # no-op that optimizer will remove
            optional_cast = f"input_f = Identity ({self.input_names[0]})"

        byte_to_float_graph = onnx.parser.parse_graph(f'''\
            byte_to_float (uint8[{input_shape_str}] {self.input_names[0]}) 
                => (float[{input_shape_str}] {self.output_names[0]})
            {{
                f_255 = Constant <value = float[1] {{255.0}}>()

                {optional_cast}
                {self.output_names[0]} = Div(input_f, f_255)
            }}
            ''')

        onnx.checker.check_graph(byte_to_float_graph)
        return byte_to_float_graph


class FloatToImageBytes(Step):
    """
    Reverse ImageBytesToFloat by converting floating point values to uint8.
    Typically this converts from range 0..1 to 0..255 but can also be used to just convert the type with rounding
    so the output is in the range 0..255
    """
    def __init__(self, multiplier: float = 255.0, name: str = None):
        super().__init__(['float_data'], ['pixel_data'], name)
        self._multiplier = multiplier

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        assert(input_type_str == 'float')

        float_to_byte_graphs = onnx.parser.parse_graph(f'''\
            float_to_type (float[{input_shape_str}] {self.input_names[0]}) 
                => (uint8[{input_shape_str}] {self.output_names[0]})
            {{
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}>()
                f_multiplier = Constant <value = float[1] {{{self._multiplier}}}> ()
                
                scaled_input = Mul ({self.input_names[0]}, f_multiplier)
                rounded = Round (scaled_input)
                clipped = Clip (rounded, f_0, f_255)
                {self.output_names[0]} = Cast <to={int(onnx.TensorProto.UINT8)}> (clipped)
            }}
            ''')

        onnx.checker.check_graph(float_to_byte_graphs)
        return float_to_byte_graphs


class Normalize(Step):
    def __init__(self, normalization_values: List[Tuple[float, float]], layout: str = 'CHW', offset: float = None,
                 name: str = None):
        """
        Provide normalization values as pairs of mean and stddev.
        Per-channel normalization requires 3 tuples.
        If a single tuple is provided the values will be applied to all channels.

        Layout can be HWC or CHW. Set hwc_layout to False for CHW.

        Data is converted to float during normalization.
        """
        super().__init__(['data'], ['normalized_data'], name)

        # duplicate for each channel if needed
        if len(normalization_values) == 1:
            normalization_values *= 3

        assert(len(normalization_values) == 3)
        self._normalization_values = normalization_values
        self._offset = offset
        assert(layout == 'HWC' or layout == 'CHW')
        self._hwc_layout = True if layout == 'HWC' else False

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        mean0 = self._normalization_values[0][0]
        mean1 = self._normalization_values[1][0]
        mean2 = self._normalization_values[2][0]
        stddev0 = self._normalization_values[0][1]
        stddev1 = self._normalization_values[1][1]
        stddev2 = self._normalization_values[2][1]

        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        values_shape = '3' if self._hwc_layout else '3, 1, 1'

        offset = ''
        if self._offset:
            offset = f'''\
                kOffset = Constant <value = float[1] {{{self._offset}}}
                {self.output_names[0]} = Add(f_div_stddev, kOffset)
            '''
        else:
            offset = f'{self.output_names[0]} = Identity (f_div_stddev)'

        normalize_graph = onnx.parser.parse_graph(f'''\
            normalize ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => (float[{input_shape_str}] {self.output_names[0]})
            {{
                kMean = Constant <value = float[{values_shape}] {{{mean0}, {mean1}, {mean2}}}> ()
                kStddev = Constant <value = float[{values_shape}] {{{stddev0}, {stddev1}, {stddev2}}}> ()                
                f_input = Cast <to = 1> ({self.input_names[0]})
                f_sub_mean = Sub (f_input, kMean)
                f_div_stddev = Div (f_sub_mean, kStddev)
                {offset}
            }}
            ''')

        onnx.checker.check_graph(normalize_graph)
        return normalize_graph


class Unsqueeze(Step):
    def __init__(self, axes: List[int], name: str = None):
        super().__init__(['data'], ['expanded'], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')

        for idx in self._axes:
            dims.insert(idx, '1')

        output_shape_str = ','.join(dims)
        axes_strs = [str(axis) for axis in self._axes]

        unsqueeze_graph = onnx.parser.parse_graph(f'''\
            unsqueeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                axes = Constant <value = int64[{len(self._axes)}] {{{','.join(axes_strs)}}}> ()
                {self.output_names[0]} = Unsqueeze ({self.input_names[0]}, axes)
            }}
            ''')

        onnx.checker.check_graph(unsqueeze_graph)
        return unsqueeze_graph


class Squeeze(Step):
    def __init__(self, axes: List[int] = None, name: str = None):
        super().__init__(['data'], ['squeezed'], name)
        self._axes = axes

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        dims = input_shape_str.split(',')
        output_dims = [dim for idx, dim in enumerate(dims) if idx not in self._axes]
        output_shape_str = ','.join(output_dims)

        if self._axes:
            axes_strs = [str(axis) for axis in self._axes]
            graph_str = f'''\
            axes = Constant <value = int64[{len(self._axes)}] {{{','.join(axes_strs)}}}> ()
            {self.output_names[0]} = Squeeze({self.input_names[0]}, axes)
            '''
        else:
            graph_str = f"{self.output_names[0]} = Squeeze({self.input_names[0]})"

        squeeze_graph = onnx.parser.parse_graph(f'''\
            squeeze ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                {graph_str}
            }}
            ''')

        onnx.checker.check_graph(squeeze_graph)
        return squeeze_graph


class Transpose(Step):
    def __init__(self, perms: List[int], name: str = None):
        super().__init__(['X'], ['transposed'], name)
        self.perms = perms

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        perms_str = ','.join([str(idx) for idx in self.perms])
        dims = input_shape_str.split(',')
        output_dims = [dims[axis] for axis in self.perms]
        output_shape_str = ','.join(output_dims)

        transpose_graph = onnx.parser.parse_graph(f'''\
            transpose ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{output_shape_str}] {self.output_names[0]})  
            {{
                {self.output_names[0]} = Transpose <perm = [{perms_str}]> ({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(transpose_graph)
        return transpose_graph


class ChannelsLastToChannelsFirst(Transpose):
    def __init__(self, has_batch_dim: bool = False, name: str = None):
        # TODO: this could be inferred from the input shape rank
        perms = [0, 3, 1, 2] if has_batch_dim else [2, 0, 1]
        super().__init__(perms, name)


class Softmax(Step):
    def __init__(self, name: str = None):
        super().__init__(['data'], ['probabilities'], name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)

        softmax_graph = onnx.parser.parse_graph(f'''\
            softmax ({input_type_str}[{input_shape_str}] {self.input_names[0]}) 
                => ({input_type_str}[{input_shape_str}] {self.output_names[0]})
            {{
                {self.output_names[0]} = Softmax ({self.input_names[0]})
            }}
            ''')

        onnx.checker.check_graph(softmax_graph, Step._custom_op_checker_context)
        return softmax_graph


class PixelsToYCbCr(Step):
    def __init__(self, layout: str = 'BGR', name: str = None):
        super().__init__(['pixels'], ['Y', 'Cb', 'Cr'], name)
        assert(layout == 'RGB' or layout == 'BGR')
        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        # input should be uint8 data HWC
        input_dims = input_shape_str.split(',')
        assert(input_type_str == 'uint8' and len(input_dims) == 3 and input_dims[2] == '3')

        # https://en.wikipedia.org/wiki/YCbCr - note there are different 'RGB' types and YCbCr conversions
        # This is the JPEG conversion and gives accuracy that is fairly equivalent to opencv2 (when compared to PIL).
        # exact weights from https://www.itu.int/rec/T-REC-T.871-201105-I/en
        rgb_weights = np.array([[0.299, 0.587, 0.114],
                                [-0.299/1.772, -0.587/1.772, 0.500],
                                [0.500, -0.587/1.402, -0.114/1.402]], dtype=np.float32)

        bias = [0., 128., 128.]

        if self._layout == 'RGB':
            weights = rgb_weights
        else:
            weights = rgb_weights[:, ::-1]  # reverse the order of the last dim to match

        # Weights are transposed for usage in matmul.
        weights_shape = "3, 3"
        weights = ','.join([str(w) for w in weights.T.flatten()])

        bias_shape = "3"
        bias = ','.join([str(b) for b in bias])

        # each output is {h, w}. TBD if input is CHW or HWC though. Once we figure that out we could copy values from
        # the input shape
        output_shape_str = f'YCbCr_ppp_{self.step_num}_h, YCbCr_ppp_{self.step_num}_w'
        assert(input_type_str == "uint8")

        # convert to float for MatMul
        # apply weights and bias
        # round and clip so it's in the range 0..255
        # convert back to uint8
        # split into channels. shape will be {h, w, 1}
        # remove the trailing '1' so output is {h, w}
        converter_graph = onnx.parser.parse_graph(f'''\
            pixels_to_YCbCr (uint8[{input_shape_str}] {self.input_names[0]})
                => (float[{output_shape_str}] {self.output_names[0]},
                    float[{output_shape_str}] {self.output_names[1]},
                    float[{output_shape_str}] {self.output_names[2]})  
            {{
                kWeights = Constant <value = float[{weights_shape}] {{{weights}}}> ()
                kBias = Constant <value = float[{bias_shape}] {{{bias}}}> ()
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}> ()

                f_pixels = Cast <to = 1> ({self.input_names[0]})
                f_weighted = MatMul(f_pixels, kWeights)
                f_biased = Add(f_weighted, kBias)
                f_rounded = Round(f_biased)
                f_clipped = Clip (f_rounded, f_0, f_255)                                
                split_Y, split_Cb, split_Cr = Split <axis = -1>(f_clipped)
                {self.output_names[0]} = Squeeze (split_Y, i64_neg1)
                {self.output_names[1]} = Squeeze (split_Cb, i64_neg1)
                {self.output_names[2]} = Squeeze (split_Cr, i64_neg1)
            }}
            ''')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


class YCbCrToPixels(Step):
    """
    Convert uint8 or float input in range 0..255 from YCbCr to RGB or BGR
    """
    def __init__(self, layout: str = 'BGR', name: str = None):
        super().__init__(['Y', 'Cb', 'Cr'], ['bgr_data'], name)
        assert(layout == 'RGB' or layout == 'BGR')
        self._layout = layout

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str0, input_shape_str0 = self._get_input_type_and_shape_strs(graph, 0)
        input_type_str1, input_shape_str1 = self._get_input_type_and_shape_strs(graph, 1)
        input_type_str2, input_shape_str2 = self._get_input_type_and_shape_strs(graph, 2)
        assert((input_type_str0 == 'uint8' and input_type_str1 == 'uint8' and input_type_str2 == 'uint8') or
               (input_type_str0 == 'float' and input_type_str1 == 'float' and input_type_str2 == 'float'))

        assert(len(input_shape_str0.split(',')) == 2 and
               len(input_shape_str1.split(',')) == 2 and
               len(input_shape_str2.split(',')) == 2)

        output_shape_str = f'{input_shape_str0}, 3'

        # https://en.wikipedia.org/wiki/YCbCr
        # exact weights from https://www.itu.int/rec/T-REC-T.871-201105-I/en
        ycbcr_to_rbg_weights = np.array([[1, 0, 1.402],
                                         [1, -0.114*1.772/0.587, -0.299*1.402/0.587],
                                         [1, 1.772, 0]])

        # reverse 2nd and 3rd entry in each row (YCbCr to YCrCb so blue and red are flipped)
        ycbcr_to_bgr_weights = np.array([[1, 1.402, 0],
                                         [1, -0.299*1.402/0.587, -0.114*1.772/0.587],
                                         [1, 0, 1.772]])

        weights = ycbcr_to_bgr_weights if self._layout == 'BGR' else ycbcr_to_rbg_weights
        bias = [0.0, 128.0, 128.0]

        weights_shape = "3, 3"
        # transpose weights for use in matmul
        weights = ','.join([str(w) for w in weights.T.flatten()])

        bias_shape = "3"
        bias = ','.join([str(b) for b in bias])

        # unsqueeze the {h, w} inputs to add channels dim. new shape is {h, w, 1}
        # merge Y, Cb, Cr data on the new channel axis
        # convert to float to apply weights etc.
        # remove bias
        # apply weights
        # round and clip to 0..255
        # convert back to uint8.
        converter_graph = onnx.parser.parse_graph(f'''\
            YCbCr_to_RGB ({input_type_str0}[{input_shape_str0}] {self.input_names[0]},
                          {input_type_str1}[{input_shape_str1}] {self.input_names[1]},
                          {input_type_str2}[{input_shape_str2}] {self.input_names[2]}) 
                => (uint8[{output_shape_str}] {self.output_names[0]})  
            {{
                kWeights = Constant <value = float[{weights_shape}] {{{weights}}}> ()
                kBias = Constant <value = float[{bias_shape}] {{{bias}}}> ()
                f_0 = Constant <value = float[1] {{0.0}}> ()
                f_255 = Constant <value = float[1] {{255.0}}> ()
                i64_neg1 = Constant <value = int64[1] {{-1}}> ()

                Y1 = Unsqueeze({self.input_names[0]}, i64_neg1)
                Cb1 = Unsqueeze({self.input_names[1]}, i64_neg1)
                Cr1 = Unsqueeze({self.input_names[2]}, i64_neg1)                
                YCbCr = Concat <axis = -1> (Y1, Cb1, Cr1)
                f_YCbCr = Cast <to = 1> (YCbCr)
                f_unbiased = Sub (f_YCbCr, kBias)
                f_pixels = MatMul (f_unbiased, kWeights)
                f_rounded = Round (f_pixels)
                clipped = Clip (f_rounded, f_0, f_255)                                
                {self.output_names[0]} = Cast <to={int(onnx.TensorProto.UINT8)}>(clipped)                  
            }}
            ''')

        onnx.checker.check_graph(converter_graph, Step._custom_op_checker_context)
        return converter_graph


class ReverseAxis(Step):
    def __init__(self, axis: int = -1, dim_value: int = -1, name: str = None):
        super().__init__(['data'], ['data_with_reversed_axis'], name)
        self._axis = axis
        self._dim_value = dim_value

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, 0)
        input_dims = input_shape_str.split(',')
        split_dim = input_dims[self._axis]

        if split_dim.isdigit():
            dim_value = int(split_dim)
            if self._dim_value != -1:
                assert(dim_value == self._dim_value)
            else:
                self._dim_value = dim_value

        split_outs = []
        for i in range(0, self._dim_value):
            split_outs.append(f'split_out_{i}')

        reverse_graph = onnx.parser.parse_graph(f'''\
            reverse_axis ({input_type_str}[{input_shape_str}] {self.input_names[0]})
                => ({input_type_str}[{input_shape_str}] {self.output_names[0]})  
            {{
                {','.join(split_outs)} = Split <axis = {self._axis}> ({self.input_names[0]})
                {self.output_names[0]} = Concat <axis = {self._axis}> ({','.join(reversed(split_outs))})
            }}
            ''')

        onnx.checker.check_graph(reverse_graph)
        return reverse_graph


class Debug(Step):
    """
    Step that can be arbitrarily inserted after a 'real' step to make that step's outputs become graph outputs.
    This allows them to be easily debugged.
    NOTE: Depending on when the Step's output's are consumed the graph output may or may not have '_debug' as a suffix
          when they become graph outputs.
          TODO: PrePostProcessor __cleanup_graph_output_names could also hide the _debug by inserting an Identity node
                to rename so we're more consistent.
    """
    def __init__(self, num_inputs: int = 1, name: str = None):
        self._num_inputs = num_inputs
        input_names = [f'input{i}' for i in range(0, num_inputs)]
        output_names = [f'debug{i}' for i in range(0, num_inputs)]

        super().__init__(input_names, output_names, name)

    def _create_graph_for_step(self, graph: onnx.GraphProto):
        input_str = ''
        output_str = ''
        output_debug_str = ''
        nodes_str = ''

        # update output names so we preserve info from the latest input names
        self.output_names = [f'{name}_debug' for name in self.input_names]

        for i in range(0, self._num_inputs):
            input_type_str, input_shape_str = self._get_input_type_and_shape_strs(graph, i)
            if i > 0:
                input_str += ", "
                output_str += ", "
                output_debug_str += ", "
                nodes_str += '\n'

            input_str += f"{input_type_str}[{input_shape_str}] {self.input_names[i]}"
            output_str += f"{input_type_str}[{input_shape_str}] {self.output_names[i]}"
            nodes_str += f"{self.output_names[i]} = Identity({self.input_names[i]})\n"

        debug_graph = onnx.parser.parse_graph(f'''\
            debug ({input_str}) 
                => ({output_str})
            {{
                {nodes_str}
            }}
            ''')

        onnx.checker.check_graph(debug_graph)
        return debug_graph


def create_value_info_for_image_bytes(name: str):
    # create a ValueInfoProto for a buffer of bytes containing an input image. could be jpeg/png/bmp
    input_type = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.UINT8, shape=['num_bytes'])
    return onnx.helper.make_value_info(name, input_type)


class ModelSource(enum.Enum):
    PYTORCH = 0
    TENSORFLOW = 1
    OTHER = 2


def imagenet_preprocessing(model_source: ModelSource = ModelSource.PYTORCH):
    """
    Common pre-processing for an imagenet trained model.
    """

    # These utils cover both cases of typical pytorch/tensorflow pre-processing for an imagenet trained model
    # https://github.com/keras-team/keras/blob/b80dd12da9c0bc3f569eca3455e77762cf2ee8ef/keras/applications/imagenet_utils.py#L177
    if model_source == ModelSource.PYTORCH:
        normalization_params = [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)]
        offset = None
    else:
        # TF processing involves moving the data into the range -1..1 instead of 0..1.
        # ImageBytesToFloat converts to range 0..1 so we use 0.5 for the stddev to expand to 0..2
        # and provide an offset of -1 to move to -1..1. We can't use -1 for the mean as the change has to apply after
        # the division.
        normalization_params = [(0, 0.5)]
        offset = -1.0

    return [
        Resize(256),
        CenterCrop(224, 224),
        # convert to CHW and then to float. these two steps are done by torchvision.transforms.ToTensor
        ChannelsLastToChannelsFirst(),
        ImageBytesToFloat(),
        Normalize(normalization_params, offset=offset),
        Unsqueeze([0])  # add batch dim
    ]


def mobilenet(model_file: Path, output_file: Path, model_source: ModelSource = ModelSource.PYTORCH):
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_value_info_for_image_bytes('image')]

    pipeline = PrePostProcessor(inputs)

    # support user providing encoded image bytes
    preprocessing = [ConvertImageToBGR(),  # custom op to convert jpg/png to BGR (output is HWC)
                     ReverseAxis(axis=2, dim_value=3, name="BGR_to_RGB")]  # Normalization params are for RGB ordering

    # plug in default imagenet pre-processing
    preprocessing.extend(imagenet_preprocessing(model_source))

    pipeline.add_pre_processing(preprocessing)

    # for mobilenet we convert the score to probabilities with softmax
    pipeline.add_post_processing([Softmax()])

    new_model = pipeline.run(model)

    onnx.save_model(new_model, str(output_file.resolve()))


def superresolution(model_file: Path, output_file: Path):
    # TODO: There seems to be a split with some super resolution models processing RGB input and some processing
    # the Y channel after converting to YCbCr.
    # For the sake of this example implementation we do the trickier YCbCr processing as that involves joining the
    # Cb and Cr channels with the model output to create the resized image.
    # Model is from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    model = onnx.load(str(model_file.resolve(strict=True)))

    inputs = [create_value_info_for_image_bytes('image')]

    # assuming input is *CHW, infer the input sizes from the model. requires model input and output has a fixed
    # size for the input and output height and width. user would have to specify otherwise.
    model_input_shape = model.graph.input[0].type.tensor_type.shape
    model_output_shape = model.graph.output[0].type.tensor_type.shape
    assert(model_input_shape.dim[-1].HasField("dim_value"))
    assert(model_input_shape.dim[-2].HasField("dim_value"))
    assert(model_output_shape.dim[-1].HasField("dim_value"))
    assert(model_output_shape.dim[-2].HasField("dim_value"))

    w_in = model_input_shape.dim[-1].dim_value
    h_in = model_input_shape.dim[-2].dim_value
    h_out = model_output_shape.dim[-2].dim_value
    w_out = model_output_shape.dim[-1].dim_value

    # pre/post processing for https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    pipeline = PrePostProcessor(inputs)
    pipeline.add_pre_processing([
        ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout
        Resize((h_in, w_in)),
        CenterCrop(h_in, w_in),
        # this produces Y, Cb and Cr outputs. each has shape {h_in, w_in}. only Y is input to model
        PixelsToYCbCr(layout="BGR"),
        # if you inserted this Debug step here the 3 outputs from PixelsToYCbCr would also be model outputs
        # Debug(num_inputs=3),
        ImageBytesToFloat(),  # Convert Y to float in range 0..1
        Unsqueeze([0, 1], "UnsqueezeY"),    # add batch and channels dim to Y so shape is {1, 1, h_in, w_in}
    ])

    # Post-processing is complicated here. resize the Cb and Cr outputs from the pre-processing to match
    # the model output size, merge those with the Y` model output, and convert back to RGB.

    # create the Steps we need to use in the manual connections
    pipeline.add_post_processing([
        Squeeze([0, 1]),  # remove batch and channels dims from Y'
        FloatToImageBytes(name='Y_out_to_bytes'),  # convert Y' to uint8 in range 0..255

        # Verbose example with param names for IoMapEntry to clarify
        (Resize((h_out, w_out), 'HW'), [IoMapEntry(producer='PixelsToYCbCr', producer_idx=1, consumer_idx=0)]),
        # the Cb and Cr values are already in the range 0..255 so multiplier is 1. we're using the step to round
        # for accuracy (a direct Cast would just truncate) and clip (to ensure range is 0..255) the values post-Resize
        FloatToImageBytes(multiplier=1.0, name='Resized_Cb'),

        (Resize((h_out, w_out), 'HW'), [IoMapEntry('PixelsToYCbCr', 2, 0)]),
        FloatToImageBytes(multiplier=1.0, name='Resized_Cr'),

        # as we're selecting outputs from multiple previous steps we need to map them to the inputs using step names
        (YCbCrToPixels(layout="BGR"), [IoMapEntry('Y_out_to_bytes', 0, 0),  # uint8 Y' with shape {h, w}
                                       IoMapEntry('Resized_Cb', 0, 1),       # uint8 Cb'
                                       IoMapEntry('Resized_Cr', 0, 2)]),     # uint8 Cr'

        ConvertBGRToImage(image_format='png')  # jpg or png are supported
    ])

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))


def main():
    argparser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Add pre and post processing to a model.
        
        Currently supports updating:
          - super resolution with YCbCr input
          - imagenet trained mobilenet   
          
        To customize, the logic in the `mobilenet` and `superresolution` functions can be used as a guide.
        Create a pipeline and add the required pre/post processing 'Steps' in the order required. Configure 
        individual steps as needed. 
        
        The updated model will be written in the same location as the original model, with '.onnx' updated to 
        '.with_pre_post_processing.onnx'
        """,
    )

    argparser.add_argument(
        "-t", "--model_type", type=str, required=True, choices=["superresolution", "mobilenet"],
        help="Model type."
    )

    argparser.add_argument(
        "--model_source", type=str, required=False, choices=["pytorch", "tensorflow"], default="pytorch",
        help="""
        Framework that model came from. In some cases there are known differences that can be taken into account when
        adding the pre/post processing to the model. Currently this equates to choosing different normalization 
        behavior for mobilenet models.
        """
    )

    argparser.add_argument("model", type=Path, help="Provide path to ONNX model to update.")

    args = argparser.parse_args()

    model_path = args.model.resolve(strict=True)
    new_model_path = model_path.with_suffix('.with_pre_post_processing.onnx')

    if args.model_type == "mobilenet":
        source = ModelSource.PYTORCH if args.model_source == "pytorch" else "tensorflow"
        mobilenet(model_path, new_model_path, source)
    else:
        superresolution(model_path, new_model_path)


if __name__ == '__main__':
    main()
