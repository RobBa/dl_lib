include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests_backend
  backend/main.cpp
  backend/test_tensorops.cpp
  backend/test_module.cpp
  backend/test_losses.cpp
  backend/test_computational_graph.cpp
  )

target_link_libraries(unit_tests_backend PRIVATE
  gtest  
  BackendCore
)

include(GoogleTest)
gtest_discover_tests(unit_tests_backend)

if(CMAKE_CUDA_COMPILER)
  file(GLOB_RECURSE CUDA_TEST_SOURCES backend/cuda/*.cu)
  add_executable(unit_tests_cuda ${CUDA_TEST_SOURCES})

  find_package(Threads REQUIRED)
  target_link_libraries(unit_tests_cuda PRIVATE
    gtest
    Threads::Threads # platform agnostic
    BackendCore
    CUDA::cudart
  )

  set_target_properties(unit_tests_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CMAKE_CUDA_ARCHITECTURES native
  )

  find_package(CUDAToolkit REQUIRED)
  target_include_directories(unit_tests_cuda PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  gtest_discover_tests(unit_tests_cuda)
endif()

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