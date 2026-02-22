include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/python_utility"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(_core MODULE
            data_modeling/py_data_modeling.cpp
            data_modeling/py_data_modeling_util.cpp
            )
            
set_target_properties(_core PROPERTIES 
                      PREFIX ""
                      OUTPUT_NAME "_core"
                      LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_DIR})