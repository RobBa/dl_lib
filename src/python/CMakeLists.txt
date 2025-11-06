include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/gpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        #"${CMAKE_CURRENT_SOURCE_DIR}/cpu"
        #"${CMAKE_CURRENT_SOURCE_DIR}/gpu"
        )

add_library(PythonLib STATIC
            layers/layers_cpu.cpp
            )

target_include_directories(PythonLib PRIVATE ${Boost_INCLUDE_DIRS})