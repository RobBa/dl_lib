include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/../layers/gpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/layers"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(dllib MODULE
            layers/layers_cpu.cpp
            )

# don't add "lib" in front
set_target_properties(layers_cpu PROPERTIES PREFIX "")