#!/usr/bin/env python3

import argparse
import pathlib
import sys

CMAKE_FLAG_TO_OPS = {
    "OCOS_ENABLE_BERT_TOKENIZER": [
        "BasicTokenizer",
        "BertTokenizer",
        "BertTokenizerDecoder",
    ],
    "OCOS_ENABLE_BLINGFIRE": [
        "BlingFireSentenceBreaker",
    ],
    "OCOS_ENABLE_CV2": [
        "GaussianBlur",
    ],
    "OCOS_ENABLE_GPT2_TOKENIZER": [
        "GPT2Tokenizer",
    ],
    "OCOS_ENABLE_MATH": [
        "SegmentExtraction",
    ],
    "OCOS_ENABLE_RE2_REGEX": [
        "StringRegexReplace",
        "StringRegexSplitWithOffsets",
    ],
    "OCOS_ENABLE_SPM_TOKENIZER": [
        "SentencepieceTokenizer",
    ],
    "OCOS_ENABLE_TF_STRING": [
        "MaskedFill",
        "StringConcat",
        "StringECMARegexReplace",
        "StringECMARegexSplitWithOffsets",
        "StringEqual",
        "StringJoin",
        "StringLength",
        "StringLower",
        "StringMapping",
        "StringSplit",
        "StringToHashBucket",
        "StringToHashBucketFast",
        "StringToVector",
        "StringUpper",
        "VectorToString",
    ],
    "OCOS_ENABLE_VISION": [
        "DecodeImage",
        "EncodeImage",
    ],
    "OCOS_ENABLE_WORDPIECE_TOKENIZER": [
        "WordpieceTokenizer",
    ],
}


def _gen_op_to_cmake_flag():
    op_to_cmake_flag = dict()
    for cmake_flag, op_list in CMAKE_FLAG_TO_OPS.items():
        for op in op_list:
            assert op not in op_to_cmake_flag, f"Duplicate op in CMAKE_FLAG_TO_OPS: {op}"
            op_to_cmake_flag[op] = cmake_flag
    return op_to_cmake_flag


OP_TO_CMAKE_FLAG = _gen_op_to_cmake_flag()

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
GENERATED_CMAKE_CONFIG_FILE = SCRIPT_DIR.parent / "cmake" / "_selectedoplist.cmake"


def gen_cmake_oplist(opconfig_file):
    print(
        "[onnxruntime-extensions] Generating CMake config file '{}' from op config file '{}'".format(
            GENERATED_CMAKE_CONFIG_FILE, args.selected_op_config_file
        )
    )

    ext_domain = "ai.onnx.contrib"  # default_opset_domain()
    new_ext_domain = "com.microsoft.extensions"
    ext_domain_cnt = 0
    cmake_options = set()
    with open(GENERATED_CMAKE_CONFIG_FILE, "w") as f:
        print("# Auto-Generated File, please do not edit!!!", file=f)
        with open(opconfig_file, "r") as opfile:
            for _ln in opfile:
                if _ln.startswith(ext_domain) or _ln.startswith(new_ext_domain):
                    ext_domain_cnt += 1
                    items = _ln.strip().split(";")
                    if len(items) < 3:
                        raise RuntimeError("The malformed operator config file.")
                    for _op in items[2].split(","):
                        if not _op:
                            continue  # is None or ""
                        if _op not in OP_TO_CMAKE_FLAG:
                            raise RuntimeError(
                                "Cannot find the custom operator({})'s build flags, please update "
                                "the CMAKE_FLAG_TO_OPS dictionary.".format(_op)
                            )
                        if OP_TO_CMAKE_FLAG[_op] not in cmake_options:
                            cmake_options.add(OP_TO_CMAKE_FLAG[_op])
                            print('set({} ON CACHE INTERNAL "")'.format(OP_TO_CMAKE_FLAG[_op]), file=f)
        print("# End of Building the Operator CMake variables", file=f)

    if ext_domain_cnt == 0:
        print(
            "[onnxruntime-extensions] warning: lines starting with extension domains of ai.onnx.contrib or "
            "com.microsoft.extensions in operators config file is 0",
            file=sys.stderr,
        )

    print("[onnxruntime-extensions] The cmake tool file has been generated successfully.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the cmake/_selectedoplist.cmake file from the selected op config file.",
    )

    parser.add_argument(
        "selected_op_config_file",
        type=pathlib.Path,
        help="Path to selected op config file.",
    )

    args = parser.parse_args()
    args.selected_op_config_file = args.selected_op_config_file.resolve(strict=True)

    return args


if __name__ == "__main__":
    args = parse_args()
    gen_cmake_oplist(args.selected_op_config_file)
