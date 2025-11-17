cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
project(slip)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# to enable find boost, see https://stackoverflow.com/a/79147222
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
find_package(Boost REQUIRED)

# for compiled boost lib
find_package(Boost REQUIRED COMPONENTS filesystem)

if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
endif()

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

option(DOUBLE_PRECISION "Compile with double precision at cost of speed?" OFF) # TODO: not implemented yet

# include python libs
if(APPLE)
  message("Getting PythonLibs on Apple (MacOSX) path")
  set(CMAKE_FIND_FRAMEWORK NEVER)
  find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
    
  message("Python_VERSION:${Python_VERSION}")
  message("Python_LIBRARIES:${Python_LIBRARIES}")
  message("Python_INCLUDE_DIRS:${${Python_INCLUDE_DIRS}}")

  set(PYTHON_LIBRARIES ${Python_LIBRARIES})
  string(COMPARE EQUAL "${PYTHON_LIBRARIES}" "" PYTHONLIBS_EMPTY)
  if(PYTHONLIBS_EMPTY)
    message(FATAL_ERROR "Problem: PYTHON_LIBRARIES not found. Do you have Python installed on your machine?")
  endif()

  string(COMPARE EQUAL "${Python_INCLUDE_DIRS}" "" PYTHONDIRS_EMPTY)
  if(PYTHONDIRS_EMPTY)
    message("Failed to automatically find Python_INCLUDE_DIRS. Setting the PYTHON_INCLUDE_DIRS variable manually. If this crashes please adjust the following
            path to the path where Python.h resides (the one matching the found Python instance). Paths must be consistent iff multiple Python versions on machine.")
    set(PYTHON_H_PATH "/usr/local/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13")
    set(PYTHON_INCLUDE_DIRS "${PYTHON_H_PATH}")
  else()
    set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
  endif()
    #FindPython3()
  message("Apple - Using Python:${Python_VERSION_MAJOR} - Libraries:${PYTHON_LIBRARIES} - IncludeDirs: ${PYTHON_INCLUDE_DIRS}")
else()
  message("Getting PythonLibs on Linux or Windows path")
  find_package(PythonLibs REQUIRED)
endif()
include_directories(${PYTHON_INCLUDE_DIRS})
message("Using Python:${Python_VERSION_MAJOR} - Libraries:${PYTHON_LIBRARIES} - IncludeDirs: ${PYTHON_INCLUDE_DIRS}")


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

# Boost is really nice library. Much better than std::regex and as many more nice components.
# set(BOOST_INCLUDE_LIBRARIES asio regex algorithm)
# set(BOOST_ENABLE_CMAKE ON)
# FetchContent_Declare(
#   Boost
#   URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
#   URL_MD5 893b5203b862eb9bbd08553e24ff146a
# )
# FetchContent_MakeAvailable(Boost)

include_directories("${PROJECT_SOURCE_DIR}/src"
                    "${PROJECT_SOURCE_DIR}/examples")

add_subdirectory(src)
add_subdirectory(examples)

#target_include_directories(my_app PRIVATE ${Boost_INCLUDE_DIRS})
# for compiled lib
#target_link_libraries(my_app PRIVATE Boost::filesystem)