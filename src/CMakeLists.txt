
add_subdirectory(backend)
add_subdirectory(python)

target_link_libraries(py_data_modeling 
                      ${Boost_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      BackendCore)

target_include_directories(py_data_modeling PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})

#target_link_libraries(py_layers PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} BackendCore)
#target_include_directories(py_layers PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# for compiled boost lib
#target_link_libraries(hello PRIVATE Boost::filesystem)