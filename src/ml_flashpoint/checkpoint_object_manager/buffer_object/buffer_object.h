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

#ifndef BUFFER_OBJECT_H_
#define BUFFER_OBJECT_H_

#include <cstddef>   // size_t
#include <optional>  // truncate_size
#include <string>    // object_id

class BufferObject {
 public:
  // This constructor creates a new buffer object with the specified ID and
  // capacity. By default, it fails if a file with the same `object_id` already
  // exists. To overwrite an existing file, set `overwrite` to true.
  explicit BufferObject(const std::string& buffer_id, size_t capacity,
                        bool overwrite = false);

  // Defines the modes for opening an existing buffer.

  // This constructor opens an existing file and memory-maps it in read-only
  // mode. The buffer's capacity is determined by the size of the file itself.
  explicit BufferObject(const std::string& object_id);

  // Destructor: Ensures resources are always released when the object goes out
  // of scope.
  ~BufferObject() noexcept;

  // Prevent copying - a BufferObject represents an actual block of memory
  // with an ID, which cannot be duplicated as-is.
  // A copy would require a new ID and memory region. Hence preventing copying
  // to avoid confusion.
  BufferObject(const BufferObject&) = delete;
  BufferObject& operator=(const BufferObject&) = delete;

  // Support moving.
  BufferObject(BufferObject&& other) noexcept;
  BufferObject& operator=(BufferObject&& other) noexcept;

  // Returns the buffer's identifier, often a fully qualified path.
  std::string get_id() const;

  // Returns the size of the buffer.
  size_t get_capacity() const;

  // Gets a pointer to the buffer's memory.
  void* get_data_ptr() const;

  // Checks if the buffer is currently in a closed state. A buffer is considered
  // closed if: (1)The close() method was explicitly or implicitly called.
  // (2)Any of its critical internal resources (like the file descriptor or
  // memory pointer) are invalid.
  bool is_closed() const;

  // Checks if the buffer's memory is mapped as read-only.
  bool is_readonly() const;

  // close the buffer object, optionally truncates the file before closing
  void close(std::optional<size_t> truncate_size = std::nullopt) noexcept;

  // Resizes the buffer to the new capacity.
  void resize(size_t new_capacity);

 private:
  std::string object_id_;  // The unique identifier of the buffer, typically the
                           // filename.
  void* data_ptr_;         // Pointer to the buffer object.
  int mmap_prot_;          // Protection flags of the mmap.
  size_t capacity_;        // The capacity (size) of the buffer in bytes.
  bool closed_;  // A boolean flag to mark if the object has been closed.
  int fd_;       // The file descriptor of the buffer object.
};

#endif
