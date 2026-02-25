/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BUFFER_HELPER_H_
#define BUFFER_HELPER_H_

#include <cstddef>  // For size_t
#include <optional>
#include <string>

#include "absl/status/status.h"

// This namespace holds low-level helper functions that directly wrap system
// calls.
namespace ml_flashpoint::checkpoint_object_manager::buffer_object::internal {

// Converts an errno value to a human-readable string.
std::string GetErrnoString(int err_num);

// Creates an absl::Status from the global errno variable.
absl::Status ErrnoToStatus(const std::string& message);

// Creates a new file, sets its size, and memory-maps it.
absl::Status create_file_and_mmap(const std::string& object_id, size_t size,
                                  int& out_fd, size_t& out_data_size,
                                  void*& out_data_ptr, bool overwrite = false);

// Opens an existing file and memory-maps it in read-only mode.
absl::Status open_file_and_mmap_ro(const std::string& object_id, int& out_fd,
                                   size_t& out_data_size, void*& out_data_ptr);

// Unmaps memory, optionally truncates the file, and closes the file descriptor.
absl::Status unmap_and_close(int fd, void* data_ptr, size_t data_size,
                             std::optional<size_t> truncate_size);
// Resizes the file to `new_size` and remaps it into memory,
// updating `data_ptr` to point to the resized memory buffer.
// If new_size == curr_size, this is a no-op and `data_ptr` is returned as is.
absl::Status resize_mmap(int fd, size_t new_size, void*& data_ptr,
                         size_t& curr_size);

};  // namespace
    // ml_flashpoint::checkpoint_object_manager::buffer_object::internal

#endif