# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import enum
import onnx

from .pre_post_processing import *


class NLPTaskType(enum.Enum):
    TokenClassification = enum.auto()
    QuestionAnswering = enum.auto()
    SequenceClassification = enum.auto()
    NextSentencePrediction = enum.auto()


class TokenizerType(enum.Enum):
    BertTokenizer = enum.auto()
    SentencePieceTokenizer = enum.auto()


def transformers_and_bert(
    input_model_file: Path,
    output_model_file: Path,
    vocab_file: Path,
    tokenizer_type: Union[TokenizerType, str],
    task_type: Union[NLPTaskType, str],
    onnx_opset: int = 16,
    add_debug_before_postprocessing=False,
):
    """construct the pipeline for a end2end model with pre and post processing. The final model can take text as inputs
    and output the result in text format for model like QA.

    Args:
        input_model_file (Path): the model file needed to be updated.
        output_model_file (Path): where to save the final onnx model.
        vocab_file (Path): the vocab file for the tokenizer.
        task_type (Union[NLPTaskType, str]): the task type of the model.
        onnx_opset (int, optional): the opset version to use. Defaults to 16.
        add_debug_before_postprocessing (bool, optional): whether to add a debug step before post processing. 
            Defaults to False.
    """
    if isinstance(task_type, str):
        task_type = NLPTaskType[task_type]
    if isinstance(tokenizer_type, str):
        tokenizer_type = TokenizerType[tokenizer_type]

    onnx_model = onnx.load(str(input_model_file.resolve(strict=True)))
    # hardcode batch size to 1
    inputs = [create_named_value("input_text", onnx.TensorProto.STRING, [1, "num_sentences"])]

    pipeline = PrePostProcessor(inputs, onnx_opset)
    tokenizer_args = TokenizerParam(
        vocab_or_file=vocab_file,
        do_lower_case=True,
        tweaked_bos_id=0,
        is_sentence_pair=True if task_type in [NLPTaskType.QuestionAnswering,
                                               NLPTaskType.NextSentencePrediction] else False,
    )

    preprocessing = [
        SentencePieceTokenizer(tokenizer_args)
        if tokenizer_type == TokenizerType.SentencePieceTokenizer else BertTokenizer(tokenizer_args),
        # uncomment this line to debug
        # Debug(2),
    ]

    # For verify results with out postprocessing
    postprocessing = [Debug()] if add_debug_before_postprocessing else []
    if task_type == NLPTaskType.QuestionAnswering:
        postprocessing.append((BertTokenizerQADecoder(tokenizer_args), [
            # input_ids
            utils.IoMapEntry("BertTokenizer", producer_idx=0, consumer_idx=2)]))
    elif task_type == NLPTaskType.SequenceClassification:
        postprocessing.append(ArgMax())
    # the other tasks don't need postprocessing or we don't support it yet.

    pipeline.add_pre_processing(preprocessing)
    pipeline.add_post_processing(postprocessing)

    new_model = pipeline.run(onnx_model)
    onnx.save_model(new_model, str(output_model_file.resolve()))
