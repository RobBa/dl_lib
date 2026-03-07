file(GLOB_RECURSE CORE_SOURCES
    activation_functions/*.cpp
    computational_graph/*.cpp
    data_modeling/*.cpp
    layers/*.cpp
    #networks/*.cpp
    training/*.cpp
    utility/*.cpp
    system/*.cpp
)

add_library(BackendCore STATIC ${CORE_SOURCES})

target_include_directories(BackendCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)