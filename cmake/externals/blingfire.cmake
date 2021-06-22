FetchContent_Declare(
        Blingfire
        GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
        GIT_TAG master
)


FetchContent_GetProperties(Blingfire)

if (NOT blingfire_POPULATED)
    FetchContent_Populate(Blingfire)

    # enable size optimization build
    add_subdirectory(${blingfire_SOURCE_DIR} ${blingfire_BINARY_DIR} EXCLUDE_FROM_ALL)
endif ()