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
)

target_link_libraries(unit_tests_backend PRIVATE
    gtest_main # pre-built main, avoids boilerplate if no custom initialization needed
    BackendCore
)

include(GoogleTest)
gtest_discover_tests(unit_tests_backend)

#find_package(Python3 COMPONENTS Interpreter)
#if(Python3_FOUND)
#    add_test(NAME python_tests
#        COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python
#        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#    )
#endif()