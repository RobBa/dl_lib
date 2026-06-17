file(GLOB_RECURSE CORE_SOURCES
    computational_graph/*.cpp
    data_modeling/*.cpp
    module/*.cpp
    shared/*.cpp
    system/*.cpp
    training/*.cpp
    utility/*.cpp
)

if(CUDA AND CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES
        computational_graph/*.cu
        data_modeling/*.cu
        module/*.cu
        shared/*.cu
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

if(CUDA AND CMAKE_CUDA_COMPILER)
    set_target_properties(BackendCore PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        # nvidia-smi --query-gpu=compute_cap --format=csv,noheader 
        # I get 12.0, hence 120
        #set(CMAKE_CUDA_ARCHITECTURES "75;86;89;100;120")
        CMAKE_CUDA_ARCHITECTURES native
    )

    find_package(CUDAToolkit REQUIRED)
    target_include_directories(BackendCore PRIVATE 
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(BackendCore
        CUDA::cudart
        CUDA::curand
    )

    if(CUDA_BACKEND STREQUAL "cuBLAS")
        message(STATUS "CUDA backend: cuBLAS")
        target_compile_definitions(BackendCore PRIVATE CUDA_BACKEND_CUBLAS)
        target_link_libraries(BackendCore CUDA::cublas)
    else()
        message(STATUS "CUDA backend: Custom")
    endif()
endif()

