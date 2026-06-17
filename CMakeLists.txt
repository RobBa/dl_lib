cmake_minimum_required(VERSION 3.24)
project(dllib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# to link shared libs. TODO: can we get rid of this, as it may cost runtime?
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYTHON_MODULE_DIR "${CMAKE_SOURCE_DIR}/python_lib/dl_lib/_compiled")

# to enable find boost, see https://stackoverflow.com/a/79147222
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost REQUIRED COMPONENTS python3)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CUDA "Enable CUDA execution for some faster data structures" ON)

if(CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    add_definitions(-D__CUDA)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CUDA_BACKEND "Custom" CACHE STRING "CUDA compute backend (Custom | cuBLAS)")
    set_property(CACHE CUDA_BACKEND PROPERTY STRINGS "Custom" "cuBLAS")
  else()
    message(WARNING "Could not find CUDA on system. Compiling without CUDA enabled")
  endif()
endif()

# On Apple, prevent cmake from preferring framework Python (causes include-path
# issues on newer macOS). On Linux/Windows the plain find_package suffices.
if(APPLE)
  set(CMAKE_FIND_FRAMEWORK NEVER)
endif()
find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
if(NOT Python_INCLUDE_DIRS)
  message(FATAL_ERROR "Python headers not found — is the Python 3 development package installed?")
endif()
include_directories(${Python_INCLUDE_DIRS})
message(STATUS "Python ${Python_VERSION} | libs: ${Python_LIBRARIES} | includes: ${Python_INCLUDE_DIRS}")

# Compiler flags. Release: optimised + strip own debug macros.
# Debug: gdb symbols + safe STL.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(
    "$<$<CONFIG:RELEASE>:-g;-UDEBUG>"
    "$<$<CONFIG:DEBUG>:-ggdb3;-D_GLIBCXX_DEBUG>"
    -Wreturn-type
  )
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fcompare-debug-second) # suppress notes in gcc output
endif()

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(src)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

option(BUILD_BENCHMARKS "Build benchmarks" OFF)
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
