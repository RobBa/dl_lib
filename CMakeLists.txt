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

#set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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

# AVX detection: combine compiler flag support with actual CPU hardware support.
# check_cxx_compiler_flag only verifies the compiler accepts the flag; it never
# runs the result, so it returns TRUE for AVX-512 even on CPUs that cannot
# execute it, causing SIGILL at runtime.
set(AVX_VERSION "AUTO" CACHE STRING "AVX-ISA to target: AUTO, AVX512, AVX2, AVX, SCALAR")
set_property(CACHE AVX_VERSION PROPERTY STRINGS AUTO AVX512 AVX2 AVX SCALAR)

include(CheckCXXCompilerFlag)

if(MSVC)
  set(AVX512_FLAG "/arch:AVX512")
  set(AVX2_FLAG   "/arch:AVX2")
  set(AVX_FLAG    "/arch:AVX")
else()
  set(AVX512_FLAG "-mavx512f;-mavx512vl;-mavx512bw;-mavx512dq")
  set(AVX2_FLAG   "-mavx2;-mfma")
  set(AVX_FLAG    "-mavx")
endif()

# Step 1: compiler accepts the flags?
check_cxx_compiler_flag("${AVX512_FLAG}" _CFLAGS_AVX512)
check_cxx_compiler_flag("${AVX2_FLAG}"   _CFLAGS_AVX2)
check_cxx_compiler_flag("${AVX_FLAG}"    _CFLAGS_AVX)

# Step 2: hardware can execute them? (Linux: /proc/cpuinfo; elsewhere: trust compiler)
set(_HW_AVX512 FALSE)
set(_HW_AVX2   FALSE)
set(_HW_AVX    FALSE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(READ "/proc/cpuinfo" _CPUINFO)
  if(_CPUINFO MATCHES "avx512f")
    set(_HW_AVX512 TRUE)
  endif()
  if(_CPUINFO MATCHES "avx2")
    set(_HW_AVX2 TRUE)
  endif()
  if(_CPUINFO MATCHES " avx ")
    set(_HW_AVX TRUE)
  endif()
else()
  set(_HW_AVX512 ${_CFLAGS_AVX512})
  set(_HW_AVX2   ${_CFLAGS_AVX2})
  set(_HW_AVX    ${_CFLAGS_AVX})
endif()

# Step 3: final variables used by configure_backend_target() — both gates must pass
set(COMPILER_SUPPORTS_AVX512 FALSE)
set(COMPILER_SUPPORTS_AVX2   FALSE)
set(COMPILER_SUPPORTS_AVX    FALSE)
if(_CFLAGS_AVX512 AND _HW_AVX512)
  set(COMPILER_SUPPORTS_AVX512 TRUE)
endif()
if(_CFLAGS_AVX2 AND _HW_AVX2)
  set(COMPILER_SUPPORTS_AVX2 TRUE)
endif()
if(_CFLAGS_AVX AND _HW_AVX)
  set(COMPILER_SUPPORTS_AVX TRUE)
endif()

# Manual override: if user pinned AVX_VERSION, honour it regardless of AUTO detection.
if(NOT AVX_VERSION STREQUAL "AUTO")
  set(COMPILER_SUPPORTS_AVX512 FALSE)
  set(COMPILER_SUPPORTS_AVX2   FALSE)
  set(COMPILER_SUPPORTS_AVX    FALSE)
  if(AVX_VERSION STREQUAL "AVX512" AND _CFLAGS_AVX512)
    set(COMPILER_SUPPORTS_AVX512 TRUE)
  endif()
  if(AVX_VERSION STREQUAL "AVX2" AND _CFLAGS_AVX2)
    set(COMPILER_SUPPORTS_AVX2 TRUE)
  endif()
  if(AVX_VERSION STREQUAL "AVX" AND _CFLAGS_AVX)
    set(COMPILER_SUPPORTS_AVX TRUE)
  endif()
endif()

if(COMPILER_SUPPORTS_AVX512)
  set(AVX_RESOLVED "AVX512")
  message(STATUS "AVX: AVX-512")
elseif(COMPILER_SUPPORTS_AVX2)
  set(AVX_RESOLVED "AVX2")
  message(STATUS "AVX: AVX2 + FMA")
elseif(COMPILER_SUPPORTS_AVX)
  set(AVX_RESOLVED "AVX")
  message(STATUS "AVX: AVX")
else()
  set(AVX_RESOLVED "SCALAR")
  message(STATUS "AVX: scalar (no SIMD)")
endif()

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
