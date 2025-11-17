include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/dtypes"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        "${CMAKE_CURRENT_SOURCE_DIR}/python"
        )

add_subdirectory(utility)
add_subdirectory(layers)
add_subdirectory(networks)
add_subdirectory(python)