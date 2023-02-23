# If the oeprator needs the cpp exceptions supports, write down their names
if (OCOS_ENABLE_GPT2_TOKENIZER)
    # gpt2 tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
    # could remove this limit when the nlohmann_json is updated in onnxruntime.
    message(FATAL_ERROR "GPT2_TOKENIZER operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_WORDPIECE_TOKENIZER)
    # wordpiece tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
    # could remove this limit when the nlohmann_json is updated in onnxruntime.
    message(FATAL_ERROR "WORDPIECE_TOKENIZER operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_BLINGFIRE)
    message(FATAL_ERROR "BLINGFIRE operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_SPM_TOKENIZER)
    message(FATAL_ERROR "SPM_TOKENIZER operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_CV2 OR OCOS_ENABLE_OPENCV_CODECS OR OCOS_ENABLE_VISION)
    message(FATAL_ERROR "the operators depending on opencv needs c++ exceptions support")
endif()
