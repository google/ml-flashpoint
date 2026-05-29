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

#ifndef BUFFER_POOL_H_
#define BUFFER_POOL_H_

#include <cstddef>
#include <string>
#include <pthread.h>

namespace ml_flashpoint::checkpoint_object_manager::buffer_object {

constexpr size_t kMaxBuffers = 64;
constexpr size_t kMaxPathLen = 256;

enum class BufferState : int {
  kFree = 0,
  kAcquired = 1,
};

struct SharedBufferInfo {
  char object_id[kMaxPathLen];
  size_t capacity;
  BufferState state;
  char associated_symlink[kMaxPathLen];
};

struct SharedBufferPoolState {
  pthread_mutex_t mutex;
  size_t num_buffers;
  size_t buffer_size;
  SharedBufferInfo buffers[kMaxBuffers];
};

class BufferPool {
 public:
  // Constructor: Initializes the pool.
  // One process must call it with initialize=true to create the shared memory.
  // Others call it with initialize=false to just attach to it.
  explicit BufferPool(const std::string& shm_name, const std::string& pool_dir = "",
                      int rank = 0, size_t num_buffers = 0,
                      size_t buffer_size = 0);
  ~BufferPool();

  // Non-copyable
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;

  // Acquires a buffer from the pool.
  // Returns the object_id (path) of the allocated buffer.
  std::string Acquire(const std::string& associated_symlink = "");

  // Releases a buffer back to the pool.
  void Release(const std::string& object_id);

  // Performs garbage collection.
  void GC();

 private:
  std::string shm_name_;
  int shm_fd_;
  SharedBufferPoolState* state_;
  bool initialized_;

  void Lock();
  void Unlock();
};

} // namespace ml_flashpoint::checkpoint_object_manager::buffer_object

#endif // BUFFER_POOL_H_
