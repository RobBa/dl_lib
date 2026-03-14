include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests_backend
    backend/test_data_modeling.cpp
    backend/test_computational_graph.cpp
    backend/test_module.cpp
    backend/test_losses.cpp
    backend/test_train_loop.cpp
    )

target_link_libraries(unit_tests_backend PRIVATE
    gtest_main # pre-built main, avoids boilerplate if no custom initialization needed
    BackendCore
)

include(GoogleTest)
gtest_discover_tests(unit_tests_backend)

find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    # replace the placeholder variables and copy resulting file in .py file
    add_test(
        NAME python_tests
        COMMAND ${Python3_EXECUTABLE} -m pytest 
                ${CMAKE_CURRENT_SOURCE_DIR}/python
    )

    # Set environment for Python to find the module
    set_tests_properties(python_tests PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PYTHON_MODULE_DIR}:$ENV{PYTHONPATH}"
    )
endif()