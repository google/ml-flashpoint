# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
project(ml_flashpoint_cpp LANGUAGES CXX)

# --- Global Build Settings ---
# Set the C++ standard for the entire project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable Position Independent Code (PIC) for all targets
# This is CRITICAL for building Python .so modules
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Code Coverage ---
option(ENABLE_COVERAGE "Enable code coverage" OFF)
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "Code coverage is enabled")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
    else()
        message(WARNING "Code coverage is only supported for GCC")
    endif()
endif()

# --- Python & Pybind ---
# Use FindPython to explicitly locate a Python interpreter and its
# development libraries, as currently recommended.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Dependencies ---
include(FetchContent)
FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20250814.1
)
FetchContent_MakeAvailable(abseil-cpp)


# --- GoogleTest ---
# Enable testing for the whole project if BUILD_TESTING is set (ON by default).
option(BUILD_TESTING "Build tests" ON)
if(BUILD_TESTING)
    enable_testing()
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
    # Include this module so child projects can use gtest_discover_tests
    include(GoogleTest)
endif()


# --- Sub-projects ---
# Add your child projects. They will inherit all settings and
# dependencies (absl::*, gtest*, pybind11::*) from this file.
add_subdirectory(src/ml_flashpoint/checkpoint_object_manager/buffer_object)
add_subdirectory(src/ml_flashpoint/checkpoint_object_manager/object_manager)
add_subdirectory(src/ml_flashpoint/replication/transfer_service)
add_subdirectory(src/ml_flashpoint/example_cpp)
