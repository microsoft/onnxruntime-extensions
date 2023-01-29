# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shutil
import re
import tempfile
import functools

from pathlib import Path

from pre_post_processing import PrePostProcessor, Debug
from pre_post_processing.steps import *
from pre_post_processing.utils import create_named_value

import transformers
import torch
import onnxruntime
import numpy as np
import onnxruntime_extensions


# avoid loading model from huggingface multiple times, it's time consuming
@functools.lru_cache
def get_tokenizer_and_huggingface_model(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    config = transformers.AutoConfig.from_pretrained(model_name)
    if model_name == "xlm-roberta-base":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        onnx_config = transformers.models.xlm_roberta.XLMRobertaOnnxConfig(config, "sequence-classification")
        text = ("Hello, my dog is cute",)
    elif model_name == "google/mobilebert-uncased":
        model = transformers.MobileBertForNextSentencePrediction.from_pretrained(model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(config, "masked-lm")
        text = ("where is Jim Henson?", "he is at school from where two blocks away")
    elif model_name == "csarron/mobilebert-uncased-squad-v2":
        model = transformers.MobileBertForQuestionAnswering.from_pretrained(model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(config, "question-answering")
        text = ("Who was Jim Henson?", "Jim Henson was a nice puppet")
    elif model_name == "lordtt13/emo-mobilebert":
        model = transformers.MobileBertForSequenceClassification.from_pretrained(model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(config, "sequence-classification")
        text = ("Hello, my dog is cute",)
    else:
        raise ValueError(f"{model_name} is not supported")
    return tokenizer, model, onnx_config, text


def export_backbone_or_use_provided(model_name: str, save_bert_onnx: bool, bert_onnx_model: Path):
    """
    To export onnx model from huggingface if user didn't provide one.
    """

    # fix the seed so we can reproduce the results
    transformers.set_seed(42)
    tokenizer, model, onnx_config, text = get_tokenizer_and_huggingface_model(model_name)
    inputs = tokenizer(*text, return_tensors="pt")

    if bert_onnx_model and bert_onnx_model.exists():
        print("User provided a customized onnx Model, skip regenerating.")
        return tokenizer, onnx.load(str(bert_onnx_model.resolve(True)))

    # tempfile will be removed automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        canonized_name = re.sub(r"[^a-zA-Z0-9]", "_", model_name) + ".onnx"
        onnx_model_path = Path(tmpdir + "/" + canonized_name)
        onnx_inputs, onnx_outputs = transformers.onnx.export(tokenizer, model, onnx_config, 16, onnx_model_path)
        if save_bert_onnx:
            shutil.copy(onnx_model_path, bert_onnx_model)
        return tokenizer, onnx.load(onnx_model_path)


def add_pre_post_to_bert(model_file: Path, save_bert_onnx: bool, model_name: str, output_file: Path):
    """construct the pipeline for a end2end model  with pre and post processing. The final model can take in text as inputs
    and output the result in text format.

    Args:
        model_file (Path): The onnx model needed to be updated, if not provided, will export from huggingface.
        save_bert_onnx (bool): To save the transformer model to onnx file if True.
        model_name (str): Which model to use in huggingface, it determinate tokenizer and onnx model backbone.
        output_file (Path): where to save the final onnx model.
    """
    # tokenizer and onnx model
    tokenizer, onnx_model = export_backbone_or_use_provided(model_name, save_bert_onnx, model_file)

    # construct graph input for different task
    if model_name in ["google/mobilebert-uncased", "csarron/mobilebert-uncased-squad-v2"]:
        inputs = [create_named_value("inputs", onnx.TensorProto.STRING, [2, "sentence_length"])]
    else:
        inputs = [create_named_value("inputs", onnx.TensorProto.STRING, ["sentence_length"])]

    pipeline = PrePostProcessor(inputs)

    # Can save tokenizer model separately
    def save_onnx(graph):
        opset_imports = [
            onnx.helper.make_operatorsetid(domain, opset)
            for domain, opset in pipeline._custom_op_checker_context.opset_imports.items()
        ]
        new_model = onnx.helper.make_model(graph, opset_imports=opset_imports)

        onnx.save_model(new_model, "debug.onnx")

    preprocessing = [
        SentencePieceTokenizer(tokenizer) if model_name == "xlm-roberta-base" else BertTokenizer(tokenizer),
        # uncomment this line to save the tokenizer model
        # Debug(custom_func=save_onnx),
    ]

    # For verify results with out postprocessing
    postprocessing = [Debug()]
    if model_name == "csarron/mobilebert-uncased-squad-v2":
        preprocessing.append(BertTokenizerQATask())
        postprocessing.append(BertTokenizerQATaskDecoder(tokenizer))
    elif model_name in ["lordtt13/emo-mobilebert", "xlm-roberta-base"]:
        postprocessing.append(SequenceClassify())

    pipeline.add_pre_processing(preprocessing)
    pipeline.add_post_processing(postprocessing)

    new_model = pipeline.run(onnx_model)

    onnx.save_model(new_model, str(output_file.resolve()))


def verify_results(model_file: Path, model_name: str, onnx_bert_model: Path = None):
    """
    Args:
        model_file: the onnx model which finalized and needs to be verified
        model_name: the huggingface model name
        onnx_bert_model: the onnx model which is generated by huggingface or user provide
    """
    tokenizer, hg_model, _, text = get_tokenizer_and_huggingface_model(model_name)
    encoded_input = tokenizer(*text, return_tensors="pt")
    transformers.set_seed(42)

    session_options = onnxruntime.SessionOptions()

    if onnx_bert_model.exists():
        session = onnxruntime.InferenceSession(str(onnx_bert_model.resolve(strict=True)))
        inputs = {key: value.detach().numpy() for key, value in encoded_input.items()}

        ref_outputs = session.run([i.name for i in session.get_outputs()], inputs)
        ref_map_out = {i.name: ref_outputs[idx] for idx, i in enumerate(session.get_outputs())}
    else:
        outs = hg_model(**encoded_input)
        ref_outputs = [out.detach().numpy() for out in list(outs.values())]
        ref_map_out = {i: ref_outputs[idx] for idx, i in enumerate(outs.keys())}

    # Load tokenizer op
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    session = onnxruntime.InferenceSession(str(model_file.resolve(strict=True)), session_options)

    # build input, QA task has 2 inputs
    if len(text) == 2:
        inputs = dict(
            inputs=np.array([[i] for i in text]),
        )
    else:
        inputs = dict(
            inputs=np.array([i for i in text]),
        )
    real_outputs = session.run([i.name for i in session.get_outputs()], inputs)
    matched_idx = [i for i, o in enumerate(session.get_outputs()) if list(ref_map_out.keys())[0] in o.name][0]

    assert np.allclose(
        real_outputs[matched_idx], ref_outputs[0], atol=1e-2
    ), f"Results do not match, expected:{ref_outputs[0]}, but got {real_outputs[matched_idx] }"
    print("Results matches:", real_outputs[0], "\ndiff:", real_outputs[matched_idx] - ref_outputs[0])


def main():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Add pre and post processing to a model.

        Currently supports updating:
          - MobileBert with different tasks
          - XLM-Roberta with classification task   
          
        This script does only server as a example of how to add pre/post processing to a transformer model.
        Usually pre-processing includes tokenizer and basic conversion of input_ids after tokenizer.
        Post-processing includes conversion of output_ids to text.
        
        A pipeline is created for organizing the required pre/post processing 'Steps' in the order required. Configure 
        individual steps as needed. 

        The updated model will be written in the same location as the original model, with '.onnx' updated to 
        '.with_pre_post_processing.onnx'
        
        Export models from huggingface is supported if there is not a onnx model provided.
        """,
    )

    parser.add_argument(
        "-t",
        "--model_type",
        type=str,
        required=True,
        choices=[
            "xlm-roberta-base",
            "google/mobilebert-uncased",
            "csarron/mobilebert-uncased-squad-v2",
            "lordtt13/emo-mobilebert",
        ],
        help="Model type.",
    )

    parser.add_argument(
        "-s",
        "--save_bert_onnx",
        type=bool,
        default=False,
        help="to save the hugging face model which only has transformer part",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="""The onnx model path which needs pre-or-post processing update 
                        or a dir-path where to save updated-model.
                        If a onnx model path is provided, the updated model will be written in the sane directory
                        with a suffix '.with_pre_post_processing.onnx'""",
    )

    args = parser.parse_args()

    model_path = args.model.resolve(strict=True)

    if model_path.is_dir():
        canonized_name = re.sub(r"[^a-zA-Z0-9]", "_", args.model_type) + ".onnx"
        model_path = model_path / canonized_name

    new_model_path = model_path.with_suffix(".with_pre_post_processing.onnx")

    add_pre_post_to_bert(model_path, args.save_bert_onnx, args.model_type, new_model_path)
    verify_results(new_model_path, args.model_type, model_path)


if __name__ == "__main__":
    main()
