
add_subdirectory(backend)
add_subdirectory(python)

target_link_libraries(_core PRIVATE
                      ${Boost_LIBRARIES} 
                      ${Python_LIBRARIES} 
                      BackendCore)

target_include_directories(_core PRIVATE 
                           ${Python_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})

target_link_libraries(_nn PRIVATE
                      ${Boost_LIBRARIES} 
                      ${Python_LIBRARIES} 
                      BackendCore)
                           
target_include_directories(_nn PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})

target_link_libraries(_sys PRIVATE
                      ${Boost_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      BackendCore)
                      
target_include_directories(_sys PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})

target_link_libraries(_train PRIVATE
                      ${Boost_LIBRARIES} 
                      ${PYTHON_LIBRARIES} 
                      BackendCore)

target_include_directories(_train PRIVATE 
                           ${PYTHON_INCLUDE_DIRS} 
                           ${Boost_INCLUDE_DIRS})