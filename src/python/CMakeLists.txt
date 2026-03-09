include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/py_utility"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(_core MODULE
            py_core/py_core.cpp
            py_core/py_core_util.cpp
            )

add_library(_network MODULE
            py_network/py_network.cpp
            #py_network/py_network_util.cpp
            )

add_library(_sys MODULE
            py_sys/py_sys.cpp
            )

set_target_properties(_core _sys _network PROPERTIES 
                      PREFIX ""
                      INSTALL_RPATH "$ORIGIN" # to find shared backend-core lib
                      BUILD_WITH_INSTALL_RPATH TRUE # use install RPATH even during build
                      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}")

set_target_properties(_core PROPERTIES 
                      OUTPUT_NAME "_core")

set_target_properties(_sys PROPERTIES 
                      OUTPUT_NAME "_sys")