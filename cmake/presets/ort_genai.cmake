# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(OCOS_ENABLE_GPT2_TOKENIZER ON CACHE INTERNAL "" FORCE)
set(OCOS_ENABLE_C_API ON CACHE INTERNAL "" FORCE)
set(OCOS_ENABLE_DLIB ON CACHE INTERNAL "" FORCE)
set(OCOS_ENABLE_VISION ON CACHE INTERNAL "" FORCE)
# OCOS_ENABLE_VENDOR_IMAGE_CODECS keeps its option default (ON: WIC on Windows, ImageIO on Apple) instead of
# being forced, so a build that can't use them (e.g. a sandboxed process where COM is blocked) can pass
# -DOCOS_ENABLE_VENDOR_IMAGE_CODECS=OFF to decode with libpng/libjpeg instead.
set(OCOS_ENABLE_MATH ON CACHE INTERNAL "" FORCE)
set(OCOS_ENABLE_AUDIO ON CACHE INTERNAL "" FORCE)

set(OCOS_ENABLE_CTEST OFF CACHE INTERNAL "" FORCE)
