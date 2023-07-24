include(ExternalProject)

ExternalProject_Add(curl7
                    PREFIX curl7
                    GIT_REPOSITORY "https://github.com/curl/curl.git"
                    GIT_TAG "curl-7_86_0"
                    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl7-src
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/curl7-build
                    CMAKE_ARGS -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DBUILD_SHARED_LIBS=OFF -DCURL_STATICLIB=ON -DHTTP_ONLY=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    INSTALL_COMMAND cmake -E echo "Skipping install step."
)

ExternalProject_Get_Property(curl7 SOURCE_DIR)
set(CURL7_SRC ${SOURCE_DIR})

ExternalProject_Get_Property(curl7 BINARY_DIR)
set(CURL7_BIN ${BINARY_DIR}/binary)
message(STATUS "CURL7_BIN: ${CURL7_BIN}")
# set(TRITON_THIRD_PARTY ${BINARY_DIR}/third-party)
