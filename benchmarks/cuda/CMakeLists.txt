find_package(CUDAToolkit REQUIRED)

add_executable(bm_matmul_cuda bm_matmul.cu)

target_link_libraries(bm_matmul_cuda
    PRIVATE
        benchmark::benchmark
        BackendCoreStatic
        CUDA::cudart
)

set_target_properties(bm_matmul_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_CUDA_ARCHITECTURES native
    RUNTIME_OUTPUT_DIRECTORY "${BENCHMARK_BIN_DIR}"
)

target_compile_options(bm_matmul_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
)
