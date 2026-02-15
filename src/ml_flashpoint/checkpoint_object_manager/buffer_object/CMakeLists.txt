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
project(buffer_object LANGUAGES CXX)

# Place the .so file directly in the source directory, to directly resolve the ext created below.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(buffer_object_lib STATIC
    buffer_object.cpp
    buffer_helper.cpp
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(buffer_object_lib PUBLIC
    absl::log
    absl::strings
    absl::status
    CUDA::cudart
    CUDA::toolkit
)

target_include_directories(buffer_object_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(buffer_object_ext bindings.cpp)

# Link the module against the static library AND the Python::Python target.
# Linking to pybind11::module automatically adds the necessary
# include directories, compiler flags, and libraries.
target_link_libraries(buffer_object_ext PUBLIC
    buffer_object_lib
    pybind11::module
)


# --- Test Target ---
if(BUILD_TESTING)
    # Include the GoogleTest module.
    include(GoogleTest)

    # Define the test source directory for readability
    set(TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/tests/checkpoint_object_manager/buffer_object)

    # Best Practice: Combine all related test source files into a single executable.
    add_executable(buffer_object_test
      ${TEST_SRC_DIR}/buffer_object_test.cpp
      ${TEST_SRC_DIR}/buffer_helper_test.cpp
    )

    # Link all necessary libraries to this unified test target just once.
    target_link_libraries(buffer_object_test PRIVATE
      buffer_object_lib      # The library we are testing
      GTest::gmock_main      # The GoogleTest library
    )

    # Discover all tests in the single test executable.
    gtest_discover_tests(buffer_object_test)
endif()
