macro(TryReEnableExceptionsIfNeed OCOS_ENABLE_CPP_EXCEPTIONS)
    # If the oeprator needs the cpp exceptions supports, write down their names
    if(OCOS_ENABLE_GPT2_TOKENIZER)
        # gpt2 tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
        # could remove this limit when the nlohmann_json is updated in onnxruntime.
        message(WARNING "GPT2_TOKENIZER operator needs c++ exceptions support, reopen exceptions")
        set(OCOS_ENABLE_CPP_EXCEPTIONS ON)
    endif()

    if(OCOS_ENABLE_WORDPIECE_TOKENIZER)
        # wordpiece tokenizer depends on nlohmann_json in onnxruntime, which is old and cannot disable exceptions.
        # could remove this limit when the nlohmann_json is updated in onnxruntime.
        message(WARNING "WORDPIECE_TOKENIZER operator needs c++ exceptions support, reopen exceptions")
        set(OCOS_ENABLE_CPP_EXCEPTIONS ON)
    endif()

    if(OCOS_ENABLE_BLINGFIRE)
        message(WARNING "BLINGFIRE operator needs c++ exceptions support, reopen exceptions")
        set(OCOS_ENABLE_CPP_EXCEPTIONS ON)
    endif()

    if(OCOS_ENABLE_SPM_TOKENIZER)
        message(WARNING "SPM_TOKENIZER operator needs c++ exceptions support, reopen exceptions")
        set(OCOS_ENABLE_CPP_EXCEPTIONS ON)
    endif()

    if(OCOS_ENABLE_CV2 OR OCOS_ENABLE_OPENCV_CODECS OR OCOS_ENABLE_VISION)
        message(WARNING "the operators depending on opencv needs c++ exceptions support, reopen exceptions")
        set(OCOS_ENABLE_CPP_EXCEPTIONS ON)
    endif()

endmacro()
