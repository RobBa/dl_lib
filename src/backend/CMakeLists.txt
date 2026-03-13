file(GLOB_RECURSE CORE_SOURCES
    computational_graph/*.cpp
    data_modeling/*.cpp
    module/*.cpp
    system/*.cpp
    training/*.cpp
    utility/*.cpp
)

add_library(BackendCore SHARED ${CORE_SOURCES})

target_include_directories(BackendCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(BackendCore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}" # make sure Python-modules see backend
)