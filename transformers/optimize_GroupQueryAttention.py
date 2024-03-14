import argparse
import logging
import torch

from onnx import ModelProto, load_model
from transformers import AutoConfig
from typing import Dict, List, Optional

from fusion_options import FusionOptions, AttentionOpType
from onnx_model_phi import PhiOnnxModel

# Map model type to tuple: optimizer class, export tools (pytorch, tf2onnx, keras2onnx), and default opt_level
MODEL_TYPES = {
    "phi": (PhiOnnxModel, "pytorch", 0),
}

def optimize_by_fusion(
    model: ModelProto,
    model_type: str = "bert",
    num_heads: int = 0,
    hidden_size: int = 0,
    optimization_options: Optional[FusionOptions] = None,
):
    """Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need to specify these parameters.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically.
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically.
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions.
                                                        Defaults to None.

     Returns:
        object of an optimizer class.
    """
    if model_type not in ["bert", "swin", "unet", "vae", "clip"] and (num_heads == 0 or hidden_size == 0):
        logger.warning(f"Please specify parameters of num_heads and hidden_size for model_type {model_type}")

    if model_type not in MODEL_TYPES:
        logger.warning(f"Unsupported model type: {model_type} for graph fusion, directly return model.")
        return OnnxModel(model)

    (optimizer_class, producer, _) = MODEL_TYPES[model_type]

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f'Model producer not matched: Expected "{producer}", Got "{model.producer_name}".'
            "Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = optimizer_class(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    optimizer.model.producer_name = "onnxruntime.transformers"
    from onnxruntime import __version__ as onnxruntime_version

    optimizer.model.producer_version = onnxruntime_version

    return optimizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_onnx_path", type=str, default="./phi2_original.onnx", help="Input ONNX model")
    parser.add_argument("--optimized_GQA_path", type=str, default="./phi2_decoder_fp16_gpu_sm8x.onnx", help="Optimized GQA model")
    args = parser.parse_args()
    return args

def optimize_to_GroupQueryAttention():
    args = parse_arguments()
    model = load_model(args.input_onnx_path)
    phi_config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True, cache_dir="./cache")
    optimization_options = FusionOptions("phi")
    optimization_options.set_attention_op_type(AttentionOpType.GroupQueryAttention)
    optimizer = optimize_by_fusion(
        model, 
        "phi", 
        num_heads=phi_config.num_attention_heads, 
        hidden_size=phi_config.hidden_size, 
        optimization_options=optimization_options)

    node_block_list = (
        [
            "Attention_29",
            "Attention_30",
            "Attention_31",
        ]
    )
    logging.info("Converting onnx model to float16/bfloat16...")
    optimizer.convert_float_to_float16(
        keep_io_types=False,
        node_block_list=node_block_list,
        use_symbolic_shape_infer=True,
        use_bfloat16_as_blocked_nodes_dtype=True,
    )
    logging.info("Converting onnx model to float16/bfloat16 done.")
    optimizer.save_model_to_file(args.optimized_GQA_path, use_external_data_format=True)

if __name__ == "__main__":
    optimize_to_GroupQueryAttention()