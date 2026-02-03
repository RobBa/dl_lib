include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/computational_graph"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_modeling"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        "${CMAKE_CURRENT_SOURCE_DIR}/system"
        "${CMAKE_CURRENT_SOURCE_DIR}/training/loss_functions"
        "${CMAKE_CURRENT_SOURCE_DIR}/training/optimizers"
        )

# compiling the libraries
add_subdirectory(computational_graph)
add_subdirectory(data_modeling)
add_subdirectory(layers)
add_subdirectory(networks)
#add_subdirectory(training)
add_subdirectory(system)
add_subdirectory(utility)

add_subdirectory(python)

target_link_libraries(py_data_modeling ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} DataModeling Utility System ComputationalGraph)
target_include_directories(py_data_modeling PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

target_link_libraries(py_layers ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Layers DataModeling Utility System ComputationalGraph)
target_include_directories(py_layers PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# for compiled boost lib
#target_link_libraries(hello PRIVATE Boost::filesystem)