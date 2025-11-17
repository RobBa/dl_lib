include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/gpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(layers_cpu SHARED
            layers/layers_cpu.cpp
            )

# don't add "lib" in front
set_target_properties(layers_cpu PROPERTIES PREFIX "")

target_link_libraries(layers_cpu ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Layers)
target_include_directories(layers_cpu PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
#target_include_directories(layers_cpu PRIVATE ${Boost_INCLUDE_DIRS})


#add_library(hello SHARED
#            hello.cpp
#            )

# don't add "lib" in front
#set_target_properties(hello PROPERTIES PREFIX "")

#target_include_directories(hello PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
#target_link_libraries(hello ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#target_link_libraries(hello PRIVATE Boost::filesystem)