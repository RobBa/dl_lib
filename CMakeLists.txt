cmake_minimum_required(VERSION 3.24)
set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
project(slip)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MINGW)
    set(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc")
endif()

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

option(DOUBLE_PRECISION "Compile with double precision at cost of speed?" OFF)

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