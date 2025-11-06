include_directories(
        #"${CMAKE_CURRENT_SOURCE_DIR}"
        #"${CMAKE_CURRENT_SOURCE_DIR}/cpu"
        #"${CMAKE_CURRENT_SOURCE_DIR}/gpu"
        )

add_library(Networks STATIC
        sequential.cpp
        )