include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}"
        )

add_library(Layers STATIC
        layer_base.cpp
        ff_layer.cpp
        )