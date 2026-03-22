file(GLOB_RECURSE CORE_SOURCES
    computational_graph/*.cpp
    data_modeling/*.cpp
    module/*.cpp
    system/*.cpp
    training/*.cpp
    utility/*.cpp
)

if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES
        computational_graph/*.cu
        data_modeling/*.cu
        module/*.cu
        system/*.cu
        training/*.cu
        utility/*.cu
    )
    list(APPEND CORE_SOURCES ${CUDA_SOURCES})
endif()

add_library(BackendCore SHARED ${CORE_SOURCES})

target_include_directories(BackendCore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(BackendCore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}" # make sure Python-modules see backend
)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(BackendCore PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # nvidia-smi --query-gpu=compute_cap --format=csv,noheader 
        # I get 12.0, hence 120
        CUDA_ARCHITECTURES 120
    )

    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(BackendCore ${CUDART_LIBRARY})
endif()