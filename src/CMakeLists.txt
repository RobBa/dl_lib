include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_modeling"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        "${CMAKE_CURRENT_SOURCE_DIR}/python"
        )

# compiling the libraries
add_subdirectory(utility)
add_subdirectory(data_modeling)
add_subdirectory(layers)
add_subdirectory(networks)
add_subdirectory(python)

target_link_libraries(layers_cpu ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Layers DataModeling Utility)
target_include_directories(layers_cpu PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# for compiled boost lib
#target_link_libraries(hello PRIVATE Boost::filesystem)