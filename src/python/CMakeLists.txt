include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers"
        "${CMAKE_CURRENT_SOURCE_DIR}/../data_modeling"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        "${CMAKE_CURRENT_SOURCE_DIR}/python_utility"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(py_data_modeling MODULE
            data_modeling/py_data_modeling.cpp
            )
set_target_properties(py_data_modeling PROPERTIES PREFIX "") # don't add "lib" in front

add_library(py_layers MODULE
            layers/py_layers.cpp
            )
set_target_properties(py_layers PROPERTIES PREFIX "") # don't add "lib" in front

#target_link_libraries(layers INTERFACE data_modeling)