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
project(transfer_service LANGUAGES CXX)

# Place the .so file directly in the source directory, to directly resolve the ext created below.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(transfer_service_lib STATIC
    transfer_service.cpp
    task.cpp
    thread_pool.cpp
    connection_pool.cpp
    net_util.cpp
    transfer_helpers.cpp
    mlf_log_sink.cpp
    ${CMAKE_SOURCE_DIR}/src/ml_flashpoint/checkpoint_object_manager/buffer_object/buffer_object.cpp
    ${CMAKE_SOURCE_DIR}/src/ml_flashpoint/checkpoint_object_manager/buffer_object/buffer_helper.cpp
)

target_link_libraries(transfer_service_lib PUBLIC absl::log absl::check absl::status absl::statusor absl::strings absl::base absl::time absl::log_initialize absl::log_globals)

target_include_directories(transfer_service_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/ml_flashpoint/checkpoint_object_manager/buffer_object
)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
#set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(transfer_service_ext bindings.cpp)

# Link the module against the static library AND the Python::Python target.
# Linking to pybind11::module automatically adds the necessary
# include directories, compiler flags, and libraries.
target_link_libraries(transfer_service_ext PUBLIC
    transfer_service_lib
    pybind11::module
)

# --- Test Target ---
if(BUILD_TESTING)
    set(TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/tests/replication/transfer_service/test)
    add_executable(transfer_service_test
        ${TEST_SRC_DIR}/protocol_test.cpp
        ${TEST_SRC_DIR}/transfer_service_test.cpp
        ${TEST_SRC_DIR}/transfer_service_p2p_test.cpp
        ${TEST_SRC_DIR}/task_test.cpp
        ${TEST_SRC_DIR}/thread_pool_test.cpp
        ${TEST_SRC_DIR}/task_queue_test.cpp
        ${TEST_SRC_DIR}/connection_pool_test.cpp
        ${TEST_SRC_DIR}/net_util_test.cpp
        ${TEST_SRC_DIR}/transfer_helpers_test.cpp
        ${TEST_SRC_DIR}/mlf_log_sink_test.cpp
    )
    target_link_libraries(transfer_service_test PRIVATE
        transfer_service_lib
        gtest
        gmock
        gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(transfer_service_test)
endif()
