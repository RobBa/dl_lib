include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/dtypes"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
        )

add_subdirectory(layers)
add_subdirectory(utility)