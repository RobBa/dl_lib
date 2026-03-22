cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
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

if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

option(CUDA "Enable CUDA execution for some faster data structures" OFF)

if (CUDA)  
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    add_definitions(-D__CUDA)
    enable_language(CUDA)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  else()
    message(WARNING "Could not find CUDA on system. Exiting.")
  endif()
endif()

# include python libs
if(APPLE)
  message("Getting PythonLibs on Apple (MacOSX) path")
  set(CMAKE_FIND_FRAMEWORK NEVER)
  find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
    
  message("Python_VERSION:${Python_VERSION}")
  message("Python_LIBRARIES:${Python_LIBRARIES}")
  message("Python_INCLUDE_DIRS:${${Python_INCLUDE_DIRS}}")

  set(PYTHON_LIBRARIES ${Python_LIBRARIES})
  string(COMPARE EQUAL "${Python_LIBRARIES}" "" PYTHONLIBS_EMPTY)
  if(PYTHONLIBS_EMPTY)
    message(FATAL_ERROR "Problem: PYTHON_LIBRARIES not found. Do you have Python installed on your machine?")
  endif()

  string(COMPARE EQUAL "${Python_INCLUDE_DIRS}" "" PYTHONDIRS_EMPTY)
  if(PYTHONDIRS_EMPTY)
    message("Failed to automatically find Python_INCLUDE_DIRS. Setting the PYTHON_INCLUDE_DIRS variable manually. If this crashes please adjust the following
            path to the path where Python.h resides (the one matching the found Python instance). Paths must be consistent iff multiple Python versions on machine.")
    set(PYTHON_H_PATH "/usr/local/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13")
    set(PYTHON_INCLUDE_DIRS "${Python_H_PATH}")
  else()
    set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
  endif()
    #FindPython3()
  message("Apple - Using Python:${Python_VERSION_MAJOR} - Libraries:${Python_LIBRARIES} - IncludeDirs: ${Python_INCLUDE_DIRS}")
else()
  message("Getting PythonLibs on Linux or Windows path")
  find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
endif()
include_directories(${Python_INCLUDE_DIRS})
message("Using Python:${Python_VERSION_MAJOR} - Libraries:${Python_LIBRARIES} - IncludeDirs: ${Python_INCLUDE_DIRS}")

#set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

# Default compiler flags:
# Set type with: -DCMAKE_BUILD_TYPE=Debug
# Default is Release, set in line 9 of this file.
# Release:
# -O3 (Be fast)
# -DNDEBUG (Disable assert)
# Debug:
# -g (Debug symbols)
# https://stackoverflow.com/a/24767451/8477066
# Extra:
# -D_GLIBCXX_DEBUG Use safe STD (with bounds checking etc).
# -ggdb3 Add extra information for gdb debugger.
# -g Also for Release, it only makes binary bigger and is still useful in practise.
# -UDEBUG Removes #ifdef DEBUG SOME_DEBUG_CODE #endif for your own debug code.
if (MSVC)
    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
      "$<$<CONFIG:RELEASE>:-g;-UDEBUG>"
      "$<$<CONFIG:DEBUG>:-ggdb3;-D_GLIBCXX_DEBUG>")
else()
    add_compile_options(
      "$<$<CONFIG:RELEASE>:-g;-UDEBUG>"
      "$<$<CONFIG:DEBUG>:-ggdb3;-D_GLIBCXX_DEBUG>"
      "-fcompare-debug-second") # This flag is only for me, to suppress notes
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wreturn-type  # Specifically catch missing returns
    )
endif()

include_directories("${PROJECT_SOURCE_DIR}/src"
                    "${PROJECT_SOURCE_DIR}/examples")

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