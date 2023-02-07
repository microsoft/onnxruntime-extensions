# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shutil
import re
import tempfile
import functools
from pathlib import Path

from .pre_post_processing import *
from .pre_post_processing.steps import *
from .pre_post_processing.utils import create_named_value

# for tokenizer
import transformers


# avoid loading model from hugging-face multiple times, it's time consuming
@functools.lru_cache
def get_tokenizer_and_model_from_huggingface(model_name):
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
        raise ValueError(f"{model_name} is not supported yet.")
    return tokenizer, model, onnx_config, text


def export_backbone(model_name: str, save_bert_onnx: bool, bert_onnx_model: Path):
    """
    To export onnx model from huggingface. This model usally has inputs "input_ids", "attention_mask", "token_type_ids",
    and has tensor outputs.
    """

    # fix the seed so we can reproduce the results
    transformers.set_seed(42)
    tokenizer, model, onnx_config, text = get_tokenizer_and_model_from_huggingface(model_name)

    if bert_onnx_model and bert_onnx_model.exists():
        print("Use cached onnx Model, skip re-exporting the backbone model.")
        return tokenizer, onnx.load(str(bert_onnx_model.resolve(True)))

    # tempfile will be removed automatically
    with tempfile.TemporaryDirectory() as tmpdir:
        canonized_name = bert_onnx_model.name
        tmp_model_path = Path(tmpdir + "/" + canonized_name)
        onnx_inputs, onnx_outputs = transformers.onnx.export(tokenizer, model, onnx_config, 16, tmp_model_path)
        if save_bert_onnx:
            shutil.copy(tmp_model_path, bert_onnx_model)
        return tokenizer, onnx.load(tmp_model_path)


def add_pre_post_to_bert(model_name: str, input_model_file: Path, save_bert_onnx: bool, output_model_file: Path):
    """construct the pipeline for a end2end model with pre and post processing. The final model can take text as inputs
    and output the result in text format for model like QA.

    Args:
        model_name (str): Which model to export in hugging-face, it determinate tokenizer and onnx model backbone.
        input_model_file (Path): The onnx model needed to be saved/cached, if not provided, will export from hugging-face.
        save_bert_onnx (bool): To save the backbone transformer model to onnx file if True.
        output_model_file (Path): where to save the final onnx model.
    """
    tokenizer, onnx_model = export_backbone(model_name, save_bert_onnx, input_model_file)

    # construct graph input for different tasks
    if model_name in ["google/mobilebert-uncased", "csarron/mobilebert-uncased-squad-v2"]:
        # if two queries required
        inputs = [create_named_value("inputs", onnx.TensorProto.STRING, [2, "sentence_length"])]
    else:
        inputs = [create_named_value("inputs", onnx.TensorProto.STRING, ["sentence_length"])]

    pipeline = PrePostProcessor(inputs)

    # Can save tokenizer model separately for debugging
    def save_onnx(graph):
        opset_imports = [
            onnx.helper.make_operatorsetid(domain, opset)
            for domain, opset in pipeline._custom_op_checker_context.opset_imports.items()
        ]
        new_model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        onnx.save_model(new_model, "debug.onnx")

    tokenizer_args = dict(
        vocab=tokenizer.vocab,
        bos_token_id=tokenizer.bos_token_id,
    )
    tokenizer_args.update(tokenizer.init_kwargs)

    if hasattr(tokenizer, 'vocab_file'):
        tokenizer_args['vocab_file'] = tokenizer.vocab_file
    if hasattr(tokenizer,'do_lower_case'):
        tokenizer_args['do_lower_case'] = tokenizer.do_lower_case
    
    preprocessing = [
        SentencePieceTokenizer(tokenizer_args)
        if model_name == "xlm-roberta-base"
        else BertTokenizer(tokenizer_args),
        # uncomment this line to save the tokenizer model
        # Debug(custom_func=save_onnx),
    ]

    # For verify results with out postprocessing
    postprocessing = [Debug()]
    if model_name == "csarron/mobilebert-uncased-squad-v2":
        preprocessing.append(BertTokenizerQATask())
        postprocessing.append(BertTokenizerQATaskDecoder(tokenizer_args))
    elif model_name in ["lordtt13/emo-mobilebert", "xlm-roberta-base"]:
        postprocessing.append(SequenceClassify())

    pipeline.add_pre_processing(preprocessing)
    pipeline.add_post_processing(postprocessing)

    new_model = pipeline.run(onnx_model)
    onnx.save_model(new_model, str(output_model_file.resolve()))


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
        
        A pipeline is created for organizing the required pre/post processing 'Steps' in the order required. 
        Configure individual steps as needed. 

        The updated model will be written in the same location as the original model, with '.onnx' updated to 
        '.with_pre_post_processing.onnx'.
        
        Export models from huggingface by default if there is not a onnx model provided.
        NOTE: if providied a onnx model, you have to make sure your model is matched with the model_type in hugging-face.
        As we are using the hugging-face tokenizer to do the pre-processing.
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
        "model_path",
        type=Path,
        help="""The onnx model path which needs pre-or-post processing update 
                        or a dir-path where to save updated-model.
                        If a onnx model path is provided, the updated model will be written in the sane directory
                        with a suffix '.with_pre_post_processing.onnx'""",
    )

    args = parser.parse_args()

    model_path = args.model_path.resolve(strict=True)
    canonized_name = re.sub(r"[^a-zA-Z0-9]", "_", args.model_type) + ".onnx"

    if model_path.is_dir():
        model_path = model_path / canonized_name

    new_model_path = model_path.with_suffix(".with_pre_post_processing.onnx")

    add_pre_post_to_bert(args.model_type, model_path, args.save_bert_onnx, new_model_path)
    print(f"model saved to {new_model_path}")
    return new_model_path


if __name__ == "__main__":
    main()
