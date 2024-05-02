include(FetchContent)
FetchContent_Declare(
        flash_attention
        GIT_REPOSITORY https://github.com/Dao-AILab/flash-attention.git
        GIT_TAG v2.3.0
)

#FetchContent_GetProperties(flash_attention)
#if(NOT flash_attention_POPULATED)
#  FetchContent_Populate(flash_attention)
#  file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_kernel.h DESTINATION ${PROJECT_SOURCE_DIR}/includes)
#endif()
FetchContent_MakeAvailable(flash_attention)
file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/utils.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/block_info.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/kernel_traits.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/softmax.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_kernel.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/philox.cuh DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
#file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/dropout.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
#file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/mask.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
#file(COPY ${flash_attention_SOURCE_DIR}/csrc/flash_attn/src/rotary.h DESTINATION ${PROJECT_SOURCE_DIR}/operators/contrib/cuda/flash_attention)
