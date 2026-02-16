#include_directories(
#        "${CMAKE_CURRENT_SOURCE_DIR}/computational_graph"
#        "${CMAKE_CURRENT_SOURCE_DIR}/data_modeling"
#        "${CMAKE_CURRENT_SOURCE_DIR}/utility"
#        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
#        "${CMAKE_CURRENT_SOURCE_DIR}/system"
#        "${CMAKE_CURRENT_SOURCE_DIR}/training/loss_functions"
#        "${CMAKE_CURRENT_SOURCE_DIR}/training/optimizers"
#        )

file(GLOB_RECURSE CORE_SOURCES
    computational_graph/*.cpp
    data_modeling/*.cpp
    #layers/*.cpp
    #networks/*.cpp
    #training/*.cpp
    utility/*.cpp
    system/*.cpp
)

add_library(BackendCore STATIC ${CORE_SOURCES})

target_include_directories(BackendCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)