# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shutil
import re
import tempfile
import functools
from pathlib import Path

from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
import onnxruntime_extensions

# for tokenizer
import transformers
import numpy as np
import onnxruntime


# avoid loading model from hugging-face multiple times, it's time consuming
@functools.lru_cache
def get_tokenizer_and_model_from_huggingface(model_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    config = transformers.AutoConfig.from_pretrained(model_name)

    if model_name == "xlm-roberta-base":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name)
        onnx_config = transformers.models.xlm_roberta.XLMRobertaOnnxConfig(
            config, "sequence-classification")
        text = ("Hello, my dog is cute",)
    elif model_name == "google/mobilebert-uncased":
        model = transformers.MobileBertForNextSentencePrediction.from_pretrained(
            model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "default")
        text = ("where is Jim Henson?","he is at school from where two blocks away")
    elif model_name == "csarron/mobilebert-uncased-squad-v2":
        model = transformers.MobileBertForQuestionAnswering.from_pretrained(
            model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "question-answering")
        text = ("Who was Jim Henson?", "Jim Henson was a nice puppet")
    elif model_name == "lordtt13/emo-mobilebert":
        model = transformers.MobileBertForSequenceClassification.from_pretrained(
            model_name)
        onnx_config = transformers.models.mobilebert.MobileBertOnnxConfig(
            config, "sequence-classification")
        text = ("Hello, my dog is cute",)
    else:
        raise ValueError(f"{model_name} is not supported yet.")
    return tokenizer, model, onnx_config, text


def export_backbone(model_name: str, bert_onnx_model: Path):
    """
    To export onnx model from huggingface. This model usually has inputs "input_ids", "attention_mask", "token_type_ids",
    and has tensor outputs.
    """

    # fix the seed so we can reproduce the results
    transformers.set_seed(42)
    tokenizer, model, onnx_config, text = get_tokenizer_and_model_from_huggingface(
        model_name)

    if bert_onnx_model and bert_onnx_model.exists():
        print("Using cached ONNX model, skipping re-exporting the backbone model.")
        return tokenizer, bert_onnx_model, onnx_config

    # tempfile will be removed automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        canonized_name = bert_onnx_model.name
        tmp_model_path = Path(tmpdir + "/" + canonized_name)
        onnx_inputs, onnx_outputs = transformers.onnx.export(
            tokenizer, model, onnx_config, 16, tmp_model_path)
        shutil.copy(tmp_model_path, bert_onnx_model)
        return tokenizer, bert_onnx_model, onnx_config


def add_pre_post_processing_to_transformers(model_name: str, input_model_file: Path, output_model_file: Path):
    """construct the pipeline for a end2end model with pre and post processing. The final model can take text as inputs
    and output the result in text format for model like QA.

    Args:
        model_name (str): Which model to export in hugging-face, it determinate tokenizer and onnx model backbone.
        input_model_file (Path): The onnx model needed to be saved/cached, if not provided, will export from hugging-face.
        output_model_file (Path): where to save the final onnx model.
    """
    tokenizer, bert_onnx_model, onnx_config = export_backbone(
        model_name, input_model_file)
    if not hasattr(tokenizer, "vocab_file"):
        vocab_file = bert_onnx_model.parent / "vocab.txt"
        import json
        with open(str(vocab_file), 'w') as f:
            f.write(json.dumps(tokenizer.vocab))
    else:
        vocab_file = tokenizer.vocab_file
    tokenizer_type = 'BertTokenizer' if model_name != 'xlm-roberta-base' else 'SentencePieceTokenizer'
    task_type = 'NextSentencePrediction' if model_name == 'google/mobilebert-uncased' else ''.join(
        [i.capitalize() for i in onnx_config.task.split('-')])
    add_ppp.transformers_and_bert(bert_onnx_model, output_model_file,
                                  vocab_file, tokenizer_type, 
                                  task_type,
                                  add_debug_before_postprocessing=True)


def verify_results_for_e2e_model(model_name: str, input_bert_model: Path, output_model_file: Path):
    """
    Args:
        output_model_file: the onnx model which finalized and needs to be verified
        model_name: the huggingface model name
        input_bert_model: the onnx model which is generated by huggingface or user provide
    """
    tokenizer, hg_model, _, text = get_tokenizer_and_model_from_huggingface(model_name)
    encoded_input = tokenizer(*text, return_tensors="pt")
    transformers.set_seed(42)

    session_options = onnxruntime.SessionOptions()

    output_name_for_verify = ''
    session = onnxruntime.InferenceSession(
        str(input_bert_model.resolve(strict=True)), providers=["CPUExecutionProvider"]
    )
    inputs = {key: value.detach().numpy()
              for key, value in encoded_input.items()}
    output_name_for_verify = session.get_outputs()[0].name
    ref_outputs = session.run([output_name_for_verify], inputs)

    # Load tokenizer op
    session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())

    session = onnxruntime.InferenceSession(
        str(output_model_file.resolve(strict=True)), session_options, providers=["CPUExecutionProvider"]
    )

    inputs = dict(
        input_text=np.array([[*text]]),
    )
    real_outputs = session.run([output_name_for_verify+"_debug"], inputs)
    assert np.allclose(
        real_outputs[0], ref_outputs[0], atol=1e-2, rtol=1e-6
    ), f"Results do not match, expected:{ref_outputs[0]}, but got {real_outputs[0] }"

    print("Results matches:",
          real_outputs[0], "\ndiff:", real_outputs[0] - ref_outputs[0])


def main():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Add pre and post processing to a model.

        This tutorial supports updating:
          - MobileBert with different tasks
          - XLM-Roberta with classification task   
          
        This tutorial provides an example of how to add pre/post processing to a transformer model.
        It can add a tokenizer (SentencePiece/Berttokenizer/HfbertTokenizer) for pre-processing,
        and a classifier/decoder for post-processing.
        
        Exports models from huggingface by default if an existing onnx model is not provided.
        NOTE: if providing a onnx model, you have to make sure your model is matched with the model_type in hugging-face as we are using the hugging-face tokenizer to do the pre-processing.
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
        "model_path",
        type=Path,
        help="""The path to an existing ONNX model or directory name to save a model exported from HuggingFace in.
                This model will be updated to add pre/post processing, and saved in the same location with the suffix 
                '.with_pre_post_processing.onnx'""",
    )

    args = parser.parse_args()

    model_path = args.model_path.resolve(strict=True)
    canonized_name = re.sub(r"[^a-zA-Z0-9]", "_", args.model_type) + ".onnx"

    if model_path.is_dir():
        model_path = model_path / canonized_name

    new_model_path = model_path.with_suffix(".with_pre_post_processing.onnx")

    add_pre_post_processing_to_transformers(args.model_type, model_path, new_model_path)
    verify_results_for_e2e_model(args.model_type, model_path, new_model_path)
    return new_model_path


if __name__ == "__main__":
    main()