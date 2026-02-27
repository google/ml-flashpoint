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
project(async_writer LANGUAGES CXX)

# --- Python & Pybind ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Torch (LibTorch) ---
# Locate Torch CMake prefix path using the current Python environment
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

add_library(async_writer_lib STATIC
    checkpoint_writer.cpp
)

target_link_libraries(async_writer_lib PUBLIC
    absl::log
    buffer_object_lib
    ${TORCH_LIBRARIES}
)

target_include_directories(async_writer_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

pybind11_add_module(async_writer_ext bindings.cpp)

target_link_libraries(async_writer_ext PUBLIC
    async_writer_lib
    pybind11::module
)
