# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(_IMGCODEC_ROOT_DIR ${dlib_SOURCE_DIR}/dlib/external)

# ----------------------------------------------------------------------------
#  project libpng
#
# ----------------------------------------------------------------------------
set (PNG_LIBRARY "libpng_static_c")
set (libPNG_SOURCE_DIR ${_IMGCODEC_ROOT_DIR}/libpng)
set (zlib_SOURCE_DIR ${_IMGCODEC_ROOT_DIR}/zlib)

if(NOT WIN32)
  find_library(M_LIBRARY
    NAMES m
    PATHS /usr/lib /usr/local/lib
  )
  if(NOT M_LIBRARY)
    message(STATUS "math lib 'libm' not found; floating point support disabled")
  endif()
else()
  # not needed on windows
  set(M_LIBRARY "")
endif()

set(lib_srcs
   ${libPNG_SOURCE_DIR}/arm/arm_init.c
   ${libPNG_SOURCE_DIR}/arm/filter_neon_intrinsics.c
   ${libPNG_SOURCE_DIR}/arm/palette_neon_intrinsics.c
   ${libPNG_SOURCE_DIR}//png.c
   ${libPNG_SOURCE_DIR}//pngerror.c
   ${libPNG_SOURCE_DIR}//pngget.c
   ${libPNG_SOURCE_DIR}//pngmem.c
   ${libPNG_SOURCE_DIR}//pngpread.c
   ${libPNG_SOURCE_DIR}//pngread.c
   ${libPNG_SOURCE_DIR}//pngrio.c
   ${libPNG_SOURCE_DIR}//pngrtran.c
   ${libPNG_SOURCE_DIR}//pngrutil.c
   ${libPNG_SOURCE_DIR}//pngset.c
   ${libPNG_SOURCE_DIR}//pngtrans.c
   ${libPNG_SOURCE_DIR}//pngwio.c
   ${libPNG_SOURCE_DIR}//pngwrite.c
   ${libPNG_SOURCE_DIR}//pngwtran.c
   ${libPNG_SOURCE_DIR}//pngwutil.c
   ${zlib_SOURCE_DIR}/adler32.c
   ${zlib_SOURCE_DIR}/compress.c
   ${zlib_SOURCE_DIR}/crc32.c
   ${zlib_SOURCE_DIR}/deflate.c
   ${zlib_SOURCE_DIR}/gzclose.c
   ${zlib_SOURCE_DIR}/gzlib.c
   ${zlib_SOURCE_DIR}/gzread.c
   ${zlib_SOURCE_DIR}/gzwrite.c
   ${zlib_SOURCE_DIR}/infback.c
   ${zlib_SOURCE_DIR}/inffast.c
   ${zlib_SOURCE_DIR}/inflate.c
   ${zlib_SOURCE_DIR}/inftrees.c
   ${zlib_SOURCE_DIR}/trees.c
   ${zlib_SOURCE_DIR}/uncompr.c
   ${zlib_SOURCE_DIR}/zutil.c
)

add_library(${PNG_LIBRARY} STATIC EXCLUDE_FROM_ALL ${lib_srcs})
target_include_directories(${PNG_LIBRARY} BEFORE PUBLIC ${zlib_SOURCE_DIR})

if(MSVC)
  target_compile_definitions(${PNG_LIBRARY} PRIVATE -D_CRT_SECURE_NO_DEPRECATE)
else()
  target_compile_options(${PNG_LIBRARY} PRIVATE -Wno-deprecated-non-prototype)
endif()

set_target_properties(${PNG_LIBRARY}
  PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      FOLDER externals)

# ----------------------------------------------------------------------------
#  project libjpeg
#
# ----------------------------------------------------------------------------
set(JPEG_LIBRARY "libjpeg_static_c")
set(libJPEG_SOURCE_DIR ${_IMGCODEC_ROOT_DIR}/libjpeg)

set(lib_srcs
  ${libJPEG_SOURCE_DIR}/jaricom.c
  ${libJPEG_SOURCE_DIR}/jcapimin.c
  ${libJPEG_SOURCE_DIR}/jcapistd.c
  ${libJPEG_SOURCE_DIR}/jcarith.c
  ${libJPEG_SOURCE_DIR}/jccoefct.c
  ${libJPEG_SOURCE_DIR}/jccolor.c
  ${libJPEG_SOURCE_DIR}/jcdctmgr.c
  ${libJPEG_SOURCE_DIR}/jchuff.c
  ${libJPEG_SOURCE_DIR}/jcinit.c
  ${libJPEG_SOURCE_DIR}/jcmainct.c
  ${libJPEG_SOURCE_DIR}/jcmarker.c
  ${libJPEG_SOURCE_DIR}/jcmaster.c
  ${libJPEG_SOURCE_DIR}/jcomapi.c
  ${libJPEG_SOURCE_DIR}/jcparam.c
  ${libJPEG_SOURCE_DIR}/jcprepct.c
  ${libJPEG_SOURCE_DIR}/jcsample.c
  ${libJPEG_SOURCE_DIR}/jdapimin.c
  ${libJPEG_SOURCE_DIR}/jdapistd.c
  ${libJPEG_SOURCE_DIR}/jdarith.c
  ${libJPEG_SOURCE_DIR}/jdatadst.c
  ${libJPEG_SOURCE_DIR}/jdatasrc.c
  ${libJPEG_SOURCE_DIR}/jdcoefct.c
  ${libJPEG_SOURCE_DIR}/jdcolor.c
  ${libJPEG_SOURCE_DIR}/jddctmgr.c
  ${libJPEG_SOURCE_DIR}/jdhuff.c
  ${libJPEG_SOURCE_DIR}/jdinput.c
  ${libJPEG_SOURCE_DIR}/jdmainct.c
  ${libJPEG_SOURCE_DIR}/jdmarker.c
  ${libJPEG_SOURCE_DIR}/jdmaster.c
  ${libJPEG_SOURCE_DIR}/jdmerge.c
  ${libJPEG_SOURCE_DIR}/jdpostct.c
  ${libJPEG_SOURCE_DIR}/jdsample.c
  ${libJPEG_SOURCE_DIR}/jerror.c
  ${libJPEG_SOURCE_DIR}/jfdctflt.c
  ${libJPEG_SOURCE_DIR}/jfdctfst.c
  ${libJPEG_SOURCE_DIR}/jfdctint.c
  ${libJPEG_SOURCE_DIR}/jidctflt.c
  ${libJPEG_SOURCE_DIR}/jidctfst.c
  ${libJPEG_SOURCE_DIR}/jidctint.c
  ${libJPEG_SOURCE_DIR}/jmemmgr.c
  ${libJPEG_SOURCE_DIR}/jmemnobs.c
  ${libJPEG_SOURCE_DIR}/jquant1.c
  ${libJPEG_SOURCE_DIR}/jquant2.c
  ${libJPEG_SOURCE_DIR}/jutils.c
  )
file(GLOB lib_hdrs ${libJPEG_SOURCE_DIR}/*.h)
add_library(${JPEG_LIBRARY} STATIC EXCLUDE_FROM_ALL ${lib_srcs} ${lib_hdrs})

if(NOT MSVC)
  set_source_files_properties(jcdctmgr.c PROPERTIES COMPILE_FLAGS "-O1")
endif()
target_compile_definitions(${JPEG_LIBRARY} PRIVATE -DNO_MKTEMP)
set_target_properties(${JPEG_LIBRARY}
  PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      FOLDER externals)
