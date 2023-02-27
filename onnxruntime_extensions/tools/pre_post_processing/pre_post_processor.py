# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx

from onnx import version_converter
from typing import List, Tuple, Union

from .utils import (
    IoMapEntry,
    IOEntryValuePreserver,
    create_custom_op_checker_context,
    sanitize_output_names,
    TENSOR_TYPE_TO_ONNX_TYPE,
)
from .step import Step


class PrePostProcessor:
    """
    Class to handle running all the pre/post processing steps and updating the model.
    """

    def __init__(self, inputs: List[onnx.ValueInfoProto] = None, onnx_opset: int = 16):
        """
        Create a PrePostProcessor instance.

        Args:
            inputs: The inputs the model will use if pre-processing is added.
            onnx_opset:  The ONNX opset to use.
                         Minimum is 16. 18 or higher is strongly preferred if image resizing is involved due to its
                         anti-aliasing ability.
        """

        if onnx_opset < 16:
            raise ValueError("ONNX opset must be 16 or later.")

        self._onnx_opset = onnx_opset
        self._custom_op_checker_context = create_custom_op_checker_context(onnx_opset)

        self.pre_processors = []
        self.post_processors = []

        # Connections for each pre/post processor. 1:1 mapping with entries in pre_processors/post_processors
        self._pre_processor_connections = []  # type: List[List[IoMapEntry]]
        self._post_processor_connections = []  # type: List[List[IoMapEntry]]

        # explicitly join outputs from Steps in pre_processors to inputs of the original model
        # format is Step or step name, step_idx, name of graph input/output
        #
        # Pre-processing we connect Step output to original model:
        #   - step_idx is for Step.output_names, and name is in graph.input
        #
        # Post-processing we connect the original model output to the Step input
        #   - step_idx is for Step.input_names, and name is in graph.output
        self._pre_processing_joins = None  # type: Union[None,List[Tuple[Union[Step, str], int, str]]]
        self._post_processing_joins = None  # type: Union[None,List[Tuple[Union[Step, str], int, str]]]

        self._inputs = inputs if inputs else []
        
        # preserve outputs from IOMapEntry, avoid it's consumed by the Follow-up steps.
        # we now can support a output value has more than one consumers with IOEntryValuePreserver.
        # IOEntryValuePreserver will preserve the output value and add it to the graph output 
        # until consumer step is done.
        self.outputs_preserver = []  # type: List[IOEntryValuePreserver]

    def add_pre_processing(self, items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]]):
        """
        Add the pre-processing steps. The last step is automatically joined to the original model inputs.

        Options are:
          Add Step with default connection of outputs from the previous step (if available) to inputs of this step.
          Add tuple of Step and list of IoMapEntry instances for manual connections to previous steps. This will be
          used to override any automatic connections.
            If IoMapEntry.producer is None it is inferred to be the immediately previous Step.
            If IoMapEntry.producer is a step name it must match the name of a previous step.
        """
        self.__add_processing(self.pre_processors, self._pre_processor_connections, items)

    def add_post_processing(self, items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]]):
        """
        Add the post-processing steps. The first step is automatically joined to the original model outputs.

        Options are:
          Add Step with default connection of outputs from the previous step (if available) to inputs of this step.
          Add tuple of Step and list of IoMapEntry instances for connections to previous steps. This will be
          used to override any automatic connections.
            If IoMapEntry.producer is None it is inferred to be the immediately previous Step.
            If IoMapEntry.producer is a step name it must match the name of a previous step.
        """
        self.__add_processing(self.post_processors, self._post_processor_connections, items)

    def _add_connection(self, consumer: Step, entry: IoMapEntry):
        producer = self.__producer_from_step_or_str(entry.producer)

        # Black does annoying things with the multi-line 'if' conditions making the code far less readable
        # fmt: off
        if not ((producer in self.pre_processors or producer in self.post_processors) and
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
        # fmt: on

        assert isinstance(producer, Step)
        consumer.connect(entry)


    def run(self, model: onnx.ModelProto):
        """
        Update the model with the graph from each step in the pre and post processing pipelines.

        Args:
            model: model to add pre/post processing to.

        Returns:
            model with pre/post processing in it.
        """

        # update the input model to the ONNX opset we're using. this is required as we implement the steps based on
        # the operator specs for this opset.
        model_opset = [
            entry.version for entry in model.opset_import if entry.domain == "" or entry.domain == "ai.onnx"
        ][0]

        if model_opset > self._onnx_opset:
            # It will probably work if the user updates PRE_POST_PROCESSING_ONNX_OPSET to match the model
            # but there are no guarantees.
            # Would only break if ONNX operators used in the pre/post processing graphs have had spec changes.
            raise ValueError(f"Model opset is {model_opset} which is newer than the opset used by this script.")
        elif model_opset < self._onnx_opset:
            model = onnx.version_converter.convert_version(model, self._onnx_opset)

        def name_nodes(new_graph: onnx.GraphProto, prefix: str):
            # simple helper so all nodes are named. this makes it far easier to debug any issues.
            idx = 0
            for n in new_graph.node:
                if not n.name:
                    n.name = prefix + str(idx)
                    idx += 1

        def preserved_apply(processor: Step, *args):
            # Trying to activate the IOEntryValuePreserver and preserve outputs.
            # and deactivate the outputs when the current graph consumed them

            for preserver in self.outputs_preserver:
                if preserver.consumer == processor:
                    preserver.is_active = False

            # IOEntryValuePreserver, preserve those outputs which has multiple consumers.
            # we explicitly add the output to the graph output.
            graph_outputs_to_maintain = [i.output for i in self.outputs_preserver if i.is_active]
            graph_for_step = processor.apply(*args, graph_outputs_to_maintain=graph_outputs_to_maintain)

            for preserver in self.outputs_preserver:
                if preserver.producer == processor:
                    preserver.is_active = True
                    preserver.output = processor.output_names[preserver.producer_idx]
            return graph_for_step

        def connect_and_run(graph: onnx.GraphProto, processor: Step, connections: List[IoMapEntry]):
            for connection in connections:
                assert connection.producer
                self._add_connection(processor, connection)

            return preserved_apply(processor, graph, self._custom_op_checker_context)

        # fix any invalid output names now if we're adding post-processing as the onnx parse_graph can't handle them
        if self.post_processors:
            sanitize_output_names(model.graph)

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
                # default to 1:1 between outputs of last step with inputs of original model
                last_step = self.pre_processors[-1]
                num_entries = min(len(last_step.output_names), len(graph.input))
                self._pre_processing_joins = [(last_step, i, graph.input[i].name) for i in range(0, num_entries)]

            # map the pre-processing outputs to graph inputs
            # we may need a natty way to get possible outputs after merge_graphs
            step_graph_outputs = [o.name for o in pre_process_graph.output]
            io_map = []  # type: List[Tuple[str, str]]
            for step, step_idx, graph_input in self._pre_processing_joins:
                io_map.append((step.output_names[step_idx], graph_input))
                step_graph_outputs.remove((step.output_names[step_idx]))

            # add outputs from previous IoMapEntry producers to maintain them as graph outputs 
            # until consumed by the final Step that requires them.
            step_graph_outputs += [
                o.name for o in graph.output if o.name not in step_graph_outputs]
            external_outputs = [
                i.output for i in self.outputs_preserver if i.is_active and i.output not in step_graph_outputs]
            if external_outputs:
                step_graph_outputs.extend(external_outputs)
            graph = onnx.compose.merge_graphs(pre_process_graph, graph, io_map, outputs=step_graph_outputs)

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
                assert graph_output in graph_outputs
                step.input_names[step_idx] = graph_output

            # create empty graph with the values that will be available to the post-processing
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

        # Make the output names nicer by removing prefixing from naming that occurred when applying the steps
        graph = PrePostProcessor.__cleanup_graph_output_names(graph)

        opset_imports = [onnx.helper.make_operatorsetid(domain, opset)
                         for domain, opset in self._custom_op_checker_context.opset_imports.items()]
        new_model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        onnx.checker.check_model(new_model)

        return new_model

    def __add_processing(
        self,
        processors: List[Step],
        processor_connections: List[List[IoMapEntry]],
        items: List[Union[Step, Tuple[Step, List[IoMapEntry]]]],
    ):
        """
        Add the pre/post processing steps and join with existing steps.

        Args:
            processors: List of processors to add items to.
            processor_connections: Populated with connections between each step. 1:1 with entries in processors.
            items: Items to add to processors.
                   Can be:
                     A Step instance. This will be implicitly joined to the immediately previous Step if one exists.
                     A tuple of (Step instance, list of IoMapEntry)
                      The IoMapEntry values are used to manually join an output from a producer Step to an input 
                      of the current Step.
                        In each IoMapEntry, if a step name is provided the producer Step will be searched for in all
                        predecessor steps. It is valid for a post-processor step to consume output from a
                        pre-processor step.
        """

        for item in items:
            step = None
            explicit_io_map_entries = None

            if isinstance(item, Step):
                step = item
            elif isinstance(item, tuple):
                step, explicit_io_map_entries = item
            else:
                raise ValueError("Unexpected type " + str(type(item)))

            # start with implicit joins and replace with explicitly provided ones
            # this allows the user to specify the minimum number of manual joins.
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
                        producer = self.__producer_from_step_or_str(entry.producer)  # throws if not found

                    io_map_entries[entry.consumer_idx] = IoMapEntry(producer, entry.producer_idx, entry.consumer_idx)
                    self.outputs_preserver.append(IOEntryValuePreserver(producer, step, entry.producer_idx))

            processors.append(step)
            processor_connections.append([entry for entry in io_map_entries if entry is not None])

    def __producer_from_step_or_str(self, entry: Union[Step, str]):
        if isinstance(entry, Step):
            return entry
        if isinstance(entry, str):
            match = (next((s for s in self.pre_processors if s.name == entry), None) or
                     next((s for s in self.post_processors if s.name == entry), None))  # fmt: skip

            if not match:
                raise ValueError(f"Step named {entry} was not found")

            return match

    @staticmethod
    def __cleanup_graph_output_names(graph: onnx.GraphProto):
        """
        Hide the prefixing of names that happens when we merge the graphs from the pre/post processing steps.
        Not essential but makes the graph outputs look far nicer.
        """

        # for each output create identity node to remove prefixing
        io_map = []
        fixes = onnx.GraphProto()

        # manually handle naming clashes
        input_names = set([i.name for i in graph.input])
        used_names = set(input_names)
        conflicts = 0

        for o in graph.output:
            if not o.name.startswith(Step.prefix):
                continue

            # we will create a small graph to do the renames so the output of the original graph will be an input
            # to that 'fixer' graph
            io_map.append((o.name, o.name))
            clean_name = o.name
            while clean_name.startswith(Step.prefix):
                # output from last step will have one prefixing stage that adds Step._prefix + '_'
                # e.g. '_ppp8_<orig_name>'
                next_underscore = clean_name.find("_", 1)
                if next_underscore > 0:
                    # this check shouldn't be necessary as we always add the trailing '_' when prefixing...
                    if len(clean_name) > next_underscore + 1:
                        next_underscore += 1
                    clean_name = clean_name[next_underscore:]

            # handle things like super resolution where there's an 'image' input and 'image' output
            if clean_name in input_names:
                clean_name += "_out"

            orig_clean_name = clean_name
            while clean_name in used_names:
                conflicts += 1
                clean_name = f"{orig_clean_name}{conflicts}"

            used_names.add(clean_name)

            renamer = onnx.helper.make_node("Identity", [o.name], [clean_name], f"Rename {o.name}")
            fixes.node.append(renamer)
            fixes.input.append(o)

            new_output = fixes.output.add()
            new_output.name = clean_name
            new_output.type.CopyFrom(o.type)

        # merge if we have any renaming to do
        if io_map:
            graph = onnx.compose.merge_graphs(graph, fixes, io_map)

        return graph
