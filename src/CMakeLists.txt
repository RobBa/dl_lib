
add_subdirectory(backend)
add_subdirectory(python)

target_link_libraries(_core PRIVATE
                      ${Boost_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      BackendCore)

target_include_directories(_core PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})

target_link_libraries(_network PRIVATE
                      ${Boost_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      BackendCore)

target_include_directories(_network PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})

target_link_libraries(_sys PRIVATE
                      ${Boost_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      BackendCore)

target_include_directories(_sys PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})