# If the oeprator needs the cpp exceptions supports, write down their names
if (OCOS_ENABLE_BLINGFIRE)
    message(FATAL_ERROR "BLINGFIRE operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_SPM_TOKENIZER)
    message(FATAL_ERROR "SPM_TOKENIZER operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_WORDPIECE_TOKENIZER)
    # wordpiece tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
    # could remove this limit when the nlohmann_json is updated in onnxruntime.
    message(FATAL_ERROR "WORDPIECE_TOKENIZER operator needs c++ exceptions support")
endif()
