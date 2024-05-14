if (OCOS_ENABLE_OPENCV_CODECS)
    # Include the subdirectories for libspng and libjpeg
    add_subdirectory(${PROJECT_SOURCE_DIR}/cmake/externals/libspng)
    add_subdirectory(${PROJECT_SOURCE_DIR}/cmake/externals/libjpeg)

    # Specify where the static libraries are built
    set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/cvout)

    # Add the libraries
    add_library(libspng STATIC IMPORTED)
    add_library(libjpeg STATIC IMPORTED)

    # Set the properties for the libraries
    set_property(TARGET libspng PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cmake/externals/cvout/libspng.a)
    set_property(TARGET libjpeg PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cmake/externals/cvout/libjpeg.a)
endif()