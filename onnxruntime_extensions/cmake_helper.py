import inspect
from ._ocos import default_opset_domain
from . import _cuops

ALL_CUSTOM_OPS = {_name: _obj for _name, _obj in inspect.getmembers(_cuops)
                  if (inspect.isclass(_obj) and issubclass(_obj, _cuops.CustomOp))}

OPMAP_TO_CMAKE_FLAGS = {'BlingFireSentenceBreaker': 'OCOS_ENABLE_BLINGFIRE',
                        'GPT2Tokenizer': 'OCOS_ENABLE_GPT2_TOKENIZER',
                        # Currently use one option for all string operators because their binary sizes are not large.
                        # Would probably split to more options like tokenizers in the future.
                        'StringConcat': 'OCOS_ENABLE_TF_STRING',
                        'StringEqual': 'OCOS_ENABLE_TF_STRING',
                        'StringToHashBucket': 'OCOS_ENABLE_TF_STRING',
                        'StringToHashBucketFast': 'OCOS_ENABLE_TF_STRING',
                        'StringJoin': 'OCOS_ENABLE_TF_STRING',
                        'StringLength': 'OCOS_ENABLE_TF_STRING',
                        'StringLower': 'OCOS_ENABLE_TF_STRING',
                        'StringRegexReplace': 'OCOS_ENABLE_TF_STRING',
                        'StringRegexSplitWithOffsets': 'OCOS_ENABLE_TF_STRING',
                        'StringSplit': 'OCOS_ENABLE_TF_STRING',
                        'StringToVector': 'OCOS_ENABLE_TF_STRING',
                        'StringUpper': 'OCOS_ENABLE_TF_STRING',
                        'VectorToString': 'OCOS_ENABLE_TF_STRING'
                        }


def gen_cmake_oplist(opconfig_file, oplist_cmake_file='_selectedoplist.cmake'):
    ext_domain = default_opset_domain()  # ai.onnx.contrib
    cmake_options = set()
    with open(oplist_cmake_file, 'w') as f:
        print("# Auto-Generated File, please do not edit manually!!!", file=f)
        with open(opconfig_file, 'r') as opfile:
            for _ln in opfile:
                if _ln.startswith(ext_domain):
                    items = _ln.strip().split(';')
                    if len(items) < 3:
                        raise RuntimeError("The malformated operator config file.")
                    for _op in items[2].split(','):
                        if not _op:
                            continue  # is None or ""
                        if _op not in OPMAP_TO_CMAKE_FLAGS:
                            raise RuntimeError(
                                "Cannot find the custom operator({})\'s build flags, please update the OPMAP_TO_CMAKE_FLAGS dictionary.".format(
                                    _op))
                        if OPMAP_TO_CMAKE_FLAGS[_op] not in cmake_options:
                            cmake_options.add(OPMAP_TO_CMAKE_FLAGS[_op])
                            print("set({} ON CACHE INTERNAL \"\")".format(OPMAP_TO_CMAKE_FLAGS[_op]), file=f)
        print("# End of Building the Operator CMake variables", file=f)

    print('The cmake tool file has been generated successfully.')
