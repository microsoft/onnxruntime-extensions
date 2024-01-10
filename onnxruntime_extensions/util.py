# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
util.py: Miscellaneous utility functions
"""

import onnx
import pathlib
import inspect

import numpy as np


# some util function for testing and tools
def get_test_data_file(*sub_dirs):
    case_file = inspect.currentframe().f_back.f_code.co_filename
    test_dir = pathlib.Path(case_file).parent
    return str(test_dir.joinpath(*sub_dirs).resolve())


def read_file(path, mode='r'):
    with open(str(path), mode) as file_content:
        return file_content.read()


def mel_filterbank(
        n_fft: int, n_mels: int = 80, sr=16000, min_mel=0, max_mel=45.245640471924965, dtype=np.float32):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns,
    and it is Slaney normalized mel-scale filterbank.
    """
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=dtype)

    # the centers of the frequency bins for the DFT
    freq_bins = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    mel = np.linspace(min_mel, max_mel, n_mels + 2)
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mel

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = np.log(6.4) / 27.0  # step size for log region

    log_t = mel >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mel[log_t] - min_log_mel))
    mel_bins = freqs

    mel_spacing = np.diff(mel_bins)

    ramps = mel_bins.reshape(-1, 1) - freq_bins.reshape(1, -1)
    for i in range(n_mels):
        left = -ramps[i] / mel_spacing[i]
        right = ramps[i + 2] / mel_spacing[i + 1]

        # intersect them with each other and zero
        fbank[i] = np.maximum(0, np.minimum(left, right))

    energy_norm = 2.0 / (mel_bins[2: n_mels + 2] - mel_bins[:n_mels])
    fbank *= energy_norm[:, np.newaxis]
    return fbank


def remove_unused_constants(subgraph):
    nodes = [_n for _n in subgraph.node]

    # Find the names of all input tensors for all nodes in the subgraph
    input_tensors = set()
    for node in nodes:
        for input_name in node.input:
            input_tensors.add(input_name)

    # Remove Constant nodes whose output is not used by any other nodes
    nodes_to_remove = []
    for node in nodes:
        if node.op_type == 'Constant':
            output_name = node.output[0]
            if output_name not in input_tensors:
                nodes_to_remove.append(node)

    for node in nodes_to_remove:
        subgraph.node.remove(node)

    # Recursively process subgraphs within this subgraph
    for node in nodes:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                remove_unused_constants(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    remove_unused_constants(subgraph)


def remove_unused_initializers(subgraph, top_level_initializers=None):
    if top_level_initializers is None:
        top_level_initializers = []
        remove_unused_constants(subgraph)
    initializers = [_i for _i in subgraph.initializer]
    nodes = subgraph.node

    # Find the names of all input tensors for all nodes in the subgraph
    input_tensors = set()
    for node in nodes:
        for input_name in node.input:
            input_tensors.add(input_name)

    # Combine top-level and current subgraph initializers
    all_initializers = initializers + top_level_initializers

    # Filter the initializers by checking if their names are in the list of used input tensors
    used_initializers = [
        init for init in all_initializers if init.name in input_tensors]

    # Update the subgraph's initializers
    del subgraph.initializer[:]
    subgraph.initializer.extend(
        [init for init in used_initializers if init in initializers])

    # Recursively process subgraphs within this subgraph
    for node in nodes:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                remove_unused_initializers(attr.g, top_level_initializers)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    remove_unused_initializers(
                        subgraph, top_level_initializers)


def quick_merge(*models, connection_indices=None):
    """
    This function merges multiple ONNX models into a single model, without performing any ONNX format checks.

    Parameters:
    *models (onnx.ModelProto): Varargs parameter representing the ONNX models to be merged.
    connection_indices (List[List[int]], optional): A nested list specifying which outputs in one model should connect 
                                                    to which inputs in the next model, based on their indices. 
                                                    If not provided, it's assumed that the sequence of outputs in 
                                                    one model exactly matches the sequence of inputs in the next model.

    Returns:
    merged_model (onnx.ModelProto): The merged ONNX model.

    Raises:
    ValueError: If there is any conflict in tensor names, either in initializers or in nodes, including subgraphs.
                If there is any conflict in opset versions for the same domain.
    """

    merged_graph = models[0].graph

    # Dictionary to store unique opsets
    opset_imports = {
        opset.domain if opset.domain else "ai.onnx": opset for opset in models[0].opset_import}

    # Iterate over all other models and merge
    for model_idx, model in enumerate(models[1:], start=1):
        if connection_indices is None:
            io_map = [(out.name, in_.name) for out, in_ in zip(
                models[model_idx - 1].graph.output, model.graph.input)]
        else:
            io_map = [(models[model_idx - 1].graph.output[out_idx].name, model.graph.input[in_idx].name)
                      for out_idx, in_idx in connection_indices[model_idx - 1]]

        merged_graph = onnx.compose.merge_graphs(merged_graph, model.graph, io_map)

        for opset in model.opset_import:
            if not opset.domain:
                opset.domain = "ai.onnx"
            if opset.domain in opset_imports and opset_imports[opset.domain].version != opset.version:
                raise ValueError(f"Conflict in opset versions for domain '{opset.domain}': " +
                                 f"model {model_idx} has version {opset.version}, while previous model has version " +
                                 f"{opset_imports[opset.domain].version}.")
            else:
                opset_imports[opset.domain] = opset

    default_opset = opset_imports.pop("ai.onnx", None)
    merged_model = onnx.helper.make_model_gen_version(merged_graph,
                                                      opset_imports=[
                                                          default_opset],
                                                      producer_name='ONNX Model Merger')
    merged_model.opset_import.extend(opset_imports.values())
    return merged_model
