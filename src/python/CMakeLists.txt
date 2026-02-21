include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/python_utility"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(py_data_modeling MODULE
            data_modeling/py_data_modeling.cpp
            data_modeling/py_data_modeling_util.cpp
            )
            
set_target_properties(py_data_modeling PROPERTIES 
                      PREFIX ""
                      OUTPUT_NAME "py_data_modeling"
                      LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_DIR})