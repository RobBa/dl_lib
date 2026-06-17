find_package(benchmark REQUIRED)

set(BENCHMARK_BIN_DIR "${CMAKE_SOURCE_DIR}/benchmarks/bin")

function(add_benchmark BENCH_NAME BENCH_SOURCE)
    add_executable(${BENCH_NAME} ${BENCH_SOURCE})
    target_link_libraries(${BENCH_NAME}
        PRIVATE
            benchmark::benchmark
            # TODO: add lib here if we benchmark it too
    )
    target_compile_options(${BENCH_NAME} PRIVATE -O2)
    set_target_properties(${BENCH_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${BENCHMARK_BIN_DIR}")
endfunction()

add_executable(bm_matmul bm_matmul.cpp)
target_link_libraries(bm_matmul
    PRIVATE
        benchmark::benchmark
        BackendCoreStatic
)
target_compile_options(bm_matmul PRIVATE -O2)
set_target_properties(bm_matmul PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${BENCHMARK_BIN_DIR}")

if(CUDA AND CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif()
