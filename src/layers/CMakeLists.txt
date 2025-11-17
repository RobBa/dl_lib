include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/gpu"
        )

add_library(Layers STATIC
        cpu/ff_layer_cpu.cpp
        )

target_link_libraries(Layers Utility)