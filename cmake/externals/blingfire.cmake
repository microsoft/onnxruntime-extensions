FetchContent_Declare(
        Blingfire
        GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
        GIT_TAG 0831265c1aca95ca02eca5bf1155e4251e545328
)


FetchContent_GetProperties(Blingfire)

if (NOT blingfire_POPULATED)
    FetchContent_Populate(Blingfire)

    # enable size optimization build
    add_subdirectory(${blingfire_SOURCE_DIR} ${blingfire_BINARY_DIR} EXCLUDE_FROM_ALL)
    # we don't use any Python code from Blingfire codebase
    set(NO_USE_FILE ${blingfire_SOURCE_DIR}/scripts/requirements.txt)
    file(TO_NATIVE_PATH ${NO_USE_FILE} NO_USE_FILE)
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        execute_process(COMMAND cmd /c "del ${NO_USE_FILE}")
    else()
        execute_process(COMMAND bash -c "rm ${NO_USE_FILE}")
    endif()
endif()
