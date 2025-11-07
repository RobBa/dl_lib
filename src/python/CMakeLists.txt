include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/gpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        #"${CMAKE_CURRENT_SOURCE_DIR}/cpu"
        #"${CMAKE_CURRENT_SOURCE_DIR}/gpu"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(layers_cpu MODULE
            layers/layers_cpu.cpp
            )

target_link_libraries(layers_cpu ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_include_directories(layers_cpu PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_include_directories(layers_cpu PRIVATE ${Boost_INCLUDE_DIRS})


#add_library(hello MODULE
#            hello.cpp
#            )

#target_link_libraries(hello ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
#target_include_directories(hello PRIVATE ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})