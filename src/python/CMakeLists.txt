include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/py_utility"
        )

# remove the lib... prefix 
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(_core MODULE
            py_core/py_core.cpp
            py_core/py_core_util.cpp
            )

add_library(_nn MODULE
            py_nn/py_nn.cpp
            )

add_library(_sys MODULE
            py_sys/py_sys.cpp
            )

add_library(_train MODULE
            py_train/py_train.cpp
            )

set_target_properties(_core _nn _sys _train PROPERTIES 
                      PREFIX ""
                      INSTALL_RPATH "$ORIGIN" # to find shared backend-core lib
                      BUILD_WITH_INSTALL_RPATH TRUE # use install RPATH even during build
                      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_MODULE_DIR}")

set_target_properties(_core PROPERTIES 
                      OUTPUT_NAME "_core")

set_target_properties(_nn PROPERTIES 
                      OUTPUT_NAME "_nn")

set_target_properties(_sys PROPERTIES 
                      OUTPUT_NAME "_sys")

set_target_properties(_train PROPERTIES 
                      OUTPUT_NAME "_train")