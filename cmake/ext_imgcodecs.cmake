# FetchContent_Declare(
#   opencv_454
#   URL           https://github.com/opencv/opencv/archive/refs/tags/4.5.4.zip
#   URL_HASH      SHA1=a60b3675763382a2c405cdb1d217b27372311c8e
#   SOURCE_SUBDIR not_set
# )

# FetchContent_MakeAvailable(opencv_454)
# set(_IMGCODEC_ROOT_DIR ${opencv_454_SOURCE_DIR}/3rdparty)
set(_IMGCODEC_ROOT_DIR ${dlib_SOURCE_DIR}/dlib/external)

set(ZLIB_LIBRARY "zlib")
set(ZLIB_SOURCE_DIR ${_IMGCODEC_ROOT_DIR}/zlib)

project(${ZLIB_LIBRARY} C)

include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckTypeSize)

#
# Check for fseeko
#
check_function_exists(fseeko HAVE_FSEEKO)
if(NOT HAVE_FSEEKO)
  add_definitions(-DNO_FSEEKO)
endif()

#
# Check for unistd.h
#
if(NOT MSVC)
  check_include_file(unistd.h Z_HAVE_UNISTD_H)
  if(Z_HAVE_UNISTD_H)
    add_definitions(-DZ_HAVE_UNISTD_H)
  endif()
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
endif()

#
# Check to see if we have large file support
#
check_type_size(off64_t OFF64_T)
if(HAVE_OFF64_T)
  add_definitions(-D_LARGEFILE64_SOURCE=1)
endif()

set(ZLIB_PUBLIC_HDRS
    ${ZLIB_SOURCE_DIR}/zconf.h
    ${ZLIB_SOURCE_DIR}/zlib.h
)
set(ZLIB_PRIVATE_HDRS
  ${ZLIB_SOURCE_DIR}/crc32.h
  ${ZLIB_SOURCE_DIR}/deflate.h
  ${ZLIB_SOURCE_DIR}/gzguts.h
  ${ZLIB_SOURCE_DIR}/inffast.h
  ${ZLIB_SOURCE_DIR}/inffixed.h
  ${ZLIB_SOURCE_DIR}/inflate.h
  ${ZLIB_SOURCE_DIR}/inftrees.h
  ${ZLIB_SOURCE_DIR}/trees.h
  ${ZLIB_SOURCE_DIR}/zutil.h
)
set(ZLIB_SRCS
  ${ZLIB_SOURCE_DIR}/adler32.c
  ${ZLIB_SOURCE_DIR}/compress.c
  ${ZLIB_SOURCE_DIR}/crc32.c
  ${ZLIB_SOURCE_DIR}/deflate.c
  ${ZLIB_SOURCE_DIR}/gzclose.c
  ${ZLIB_SOURCE_DIR}/gzlib.c
  ${ZLIB_SOURCE_DIR}/gzread.c
  ${ZLIB_SOURCE_DIR}/gzwrite.c
  ${ZLIB_SOURCE_DIR}/inflate.c
  ${ZLIB_SOURCE_DIR}/infback.c
  ${ZLIB_SOURCE_DIR}/inftrees.c
  ${ZLIB_SOURCE_DIR}/inffast.c
  ${ZLIB_SOURCE_DIR}/trees.c
  ${ZLIB_SOURCE_DIR}/uncompr.c
  ${ZLIB_SOURCE_DIR}/zutil.c
)

add_library(${ZLIB_LIBRARY} STATIC EXCLUDE_FROM_ALL ${ZLIB_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})
set_target_properties(${ZLIB_LIBRARY} PROPERTIES DEFINE_SYMBOL ZLIB_DLL)

# ----------------------------------------------------------------------------
#  CMake file for libpng. See root CMakeLists.txt
#
# ----------------------------------------------------------------------------
set (PNG_LIBRARY "libpng")
set (libPNG_SOURCE_DIR ${_IMGCODEC_ROOT_DIR}/libpng)

if(ENABLE_NEON)
  project(${PNG_LIBRARY} C ASM)
else()
  project(${PNG_LIBRARY} C)
endif()

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

file(GLOB lib_srcs ${libPNG_SOURCE_DIR}/*.c)
file(GLOB lib_hdrs ${libPNG_SOURCE_DIR}/*.h)

if(ARM OR AARCH64)
  if(ENABLE_NEON)
    if(NOT AARCH64)
      list(APPEND lib_srcs arm/filter_neon.S)
    endif()
    list(APPEND lib_srcs arm/arm_init.c arm/filter_neon_intrinsics.c arm/palette_neon_intrinsics.c)
    add_definitions(-DPNG_ARM_NEON_OPT=2)
  else()
    add_definitions(-DPNG_ARM_NEON_OPT=0) # NEON assembler is not supported
  endif()
endif()

if(";${CPU_BASELINE_FINAL};" MATCHES "SSE2"
    AND (NOT MSVC OR (MSVC_VERSION GREATER 1799))) # MSVS2013+ (issue #7232)
  list(APPEND lib_srcs intel/intel_init.c intel/filter_sse2_intrinsics.c)
  add_definitions(-DPNG_INTEL_SSE)
endif()

# set definitions and sources for MIPS
if(";${CPU_BASELINE_FINAL};" MATCHES "MSA")
    list(APPEND lib_srcs mips/mips_init.c mips/filter_msa_intrinsics.c)
    add_definitions(-DPNG_MIPS_MSA_OPT=2)
else()
    add_definitions(-DPNG_MIPS_MSA_OPT=0)
endif()

if(PPC64LE OR PPC64)
  # VSX3 features are backwards compatible
  if(";${CPU_BASELINE_FINAL};" MATCHES "VSX.*"
      AND NOT PPC64)
    list(APPEND lib_srcs powerpc/powerpc_init.c powerpc/filter_vsx_intrinsics.c)
    add_definitions(-DPNG_POWERPC_VSX_OPT=2)
  else()
    add_definitions(-DPNG_POWERPC_VSX_OPT=0)
  endif()
endif()

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

add_library(${PNG_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})
# target_compile_definitions(${PNG_LIBRARY} PRIVATE -DPNG_SIMPLIFIED_READ_SUPPORTED=1 -DPNG_SIMPLIFIED_WRITE_SUPPORTED=1)
target_link_libraries(${PNG_LIBRARY} PUBLIC ${ZLIB_LIBRARY})

# set_target_properties(${PNG_LIBRARY}
#   PROPERTIES OUTPUT_NAME ${PNG_LIBRARY}
#   DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
#   COMPILE_PDB_NAME ${PNG_LIBRARY}
#   COMPILE_PDB_NAME_DEBUG "${PNG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
#   ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
# )

set(JPEG_LIBRARY "libjpeg")
set(libJPEG_SOURCE_DIR ${_IMGCODEC_ROOT_DIR}/libjpeg)
project(${JPEG_LIBRARY})

file(GLOB lib_srcs ${libJPEG_SOURCE_DIR}/*.c)
file(GLOB lib_hdrs ${libJPEG_SOURCE_DIR}/*.h)

# if(ANDROID OR IOS OR APPLE)
#   ocv_list_filterout(lib_srcs jmemansi.c)
# else()
#   ocv_list_filterout(lib_srcs jmemnobs.c)
# endif()

# ----------------------------------------------------------------------------------
#         Define the library target:
# ----------------------------------------------------------------------------------

add_library(${JPEG_LIBRARY} STATIC ${OPENCV_3RDPARTY_EXCLUDE_FROM_ALL} ${lib_srcs} ${lib_hdrs})

if(GCC OR CLANG)
  set_source_files_properties(jcdctmgr.c PROPERTIES COMPILE_FLAGS "-O1")
endif()
target_compile_definitions(${JPEG_LIBRARY} PRIVATE -DNO_MKTEMP)

# ocv_warnings_disable(CMAKE_C_FLAGS -Wcast-align -Wshadow -Wunused -Wshift-negative-value -Wimplicit-fallthrough)
# ocv_warnings_disable(CMAKE_C_FLAGS -Wunused-parameter) # clang
# ocv_warnings_disable(CMAKE_C_FLAGS /wd4013 /wd4244 /wd4267) # vs2005

# set_target_properties(${JPEG_LIBRARY}
#   PROPERTIES OUTPUT_NAME ${JPEG_LIBRARY}
#   DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
#   COMPILE_PDB_NAME ${JPEG_LIBRARY}
#   COMPILE_PDB_NAME_DEBUG "${JPEG_LIBRARY}${OPENCV_DEBUG_POSTFIX}"
#   ARCHIVE_OUTPUT_DIRECTORY ${3P_LIBRARY_OUTPUT_PATH}
#   )