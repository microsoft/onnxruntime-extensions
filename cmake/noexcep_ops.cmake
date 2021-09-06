# If the oeprator needs the cpp exceptions supports, write down their names
if (OCOS_ENABLE_GPT2_TOKENIZER)
    # gpt2 tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
    # could remove this limit when the nlohmann_json is updated in onnxruntime.
    message(FATAL_ERROR "[onnxruntime-extensions] GPT2_TOKENIZER operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_BLINGFIRE)
    message(STATUS "[onnxruntime-extensions] BLINGFIRE operator needs c++ exceptions support, enable exceptions automatically!")
endif()
if (OCOS_ENABLE_SPM_TOKENIZER)
    message(FATAL_ERROR "[onnxruntime-extensions] SPM_TOKENIZER operator needs c++ exceptions support")
endif()
if (OCOS_ENABLE_WORDPIECE_TOKENIZER)
    # wordpiece tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
    # could remove this limit when the nlohmann_json is updated in onnxruntime.
    message(FATAL_ERROR "[onnxruntime-extensions] WORDPIECE_TOKENIZER operator needs c++ exceptions support")
endif()
