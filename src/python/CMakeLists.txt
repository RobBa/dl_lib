include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/python_utility"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(py_data_modeling MODULE
            data_modeling/py_data_modeling.cpp
            )
set_target_properties(py_data_modeling PROPERTIES PREFIX "") # don't add "lib" in front

#target_link_libraries(py_data_modeling ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} BackendCore)
#target_include_directories(py_data_modeling PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

#add_library(py_layers MODULE
#            layers/py_layers.cpp
#            )
#set_target_properties(py_layers PROPERTIES PREFIX "") # don't add "lib" in front

#target_link_libraries(layers INTERFACE data_modeling)