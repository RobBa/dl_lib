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

# Shared helper: applies all CUDA/include settings to a backend target.
function(configure_backend_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    if(CUDA AND CMAKE_CUDA_COMPILER)
        set_target_properties(${TARGET_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CMAKE_CUDA_ARCHITECTURES native
        )

        find_package(CUDAToolkit REQUIRED)
        target_include_directories(${TARGET_NAME} PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${TARGET_NAME}
            CUDA::cudart
            CUDA::curand
        )

        if(CUDA_BACKEND STREQUAL "cuBLAS")
            target_compile_definitions(${TARGET_NAME} PRIVATE CUDA_BACKEND_CUBLAS)
            target_link_libraries(${TARGET_NAME} CUDA::cublas)
        endif()
    endif()
endfunction()

add_library(BackendCore SHARED ${CORE_SOURCES})
set_target_properties(BackendCore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}" # make sure Python-modules see backend
)
configure_backend_target(BackendCore)

if(CUDA AND CMAKE_CUDA_COMPILER)
    if(CUDA_BACKEND STREQUAL "cuBLAS")
        message(STATUS "CUDA backend: cuBLAS")
    else()
        message(STATUS "CUDA backend: Custom")
    endif()
endif()

# Static variant — only built when benchmarks request it, to avoid double
# compilation in normal builds.
if(BUILD_BENCHMARKS)
    add_library(BackendCoreStatic STATIC ${CORE_SOURCES})
    configure_backend_target(BackendCoreStatic)
endif()

