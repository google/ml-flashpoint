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

#include "buffer_pool.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <filesystem>

#include "absl/log/log.h"
#include "buffer_object.h" // Needed for pre-allocation

namespace ml_flashpoint::checkpoint_object_manager::buffer_object {

BufferPool::BufferPool(const std::string& shm_name, const std::string& pool_dir,
                      int rank, size_t num_buffers,
                      size_t buffer_size)
    : shm_name_(shm_name) {
  
  // Try to create exclusively
  shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
  bool is_creator = true;

  if (shm_fd_ == -1) {
    if (errno == EEXIST) {
      // Already exists, try to open read-write
      shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
      is_creator = false;
    }
    
    if (shm_fd_ == -1) {
      throw std::runtime_error("shm_open failed: " + std::string(strerror(errno)));
    }
  }

  initialized_ = is_creator; // We use initialized_ to know if we should unlink in destructor

  size_t shm_size = sizeof(SharedBufferPoolState);
  if (is_creator) {
    if (ftruncate(shm_fd_, shm_size) == -1) {
      close(shm_fd_);
      shm_unlink(shm_name_.c_str());
      throw std::runtime_error("ftruncate failed: " + std::string(strerror(errno)));
    }
  }

  void* ptr = mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (ptr == MAP_FAILED) {
    close(shm_fd_);
    if (is_creator) {
      shm_unlink(shm_name_.c_str());
    }
    throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
  }

  state_ = static_cast<SharedBufferPoolState*>(ptr);

  if (is_creator) {
    // Initialize mutex
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&state_->mutex, &attr);
    pthread_mutexattr_destroy(&attr);

    state_->num_buffers = num_buffers;
    state_->buffer_size = buffer_size;
    
    for (size_t i = 0; i < kMaxBuffers; ++i) {
      state_->buffers[i].state = BufferState::kFree;
      state_->buffers[i].object_id[0] = '\0';
      state_->buffers[i].associated_symlink[0] = '\0';
      state_->buffers[i].capacity = 0;
    }

    for (size_t i = 0; i < num_buffers; ++i) {
      std::string buffer_name = "buffer_" + std::to_string(rank) + "_" + std::to_string(i) + ".dist";
      std::string buffer_path = (std::filesystem::path(pool_dir) / buffer_name).string();
      snprintf(state_->buffers[i].object_id, kMaxPathLen, "%s", buffer_path.c_str());
      state_->buffers[i].capacity = buffer_size;
      
      // Pre-allocate file
      try {
          BufferObject buf(buffer_path, buffer_size, true);
          LOG(INFO) << "Pre-allocated buffer file: " << buffer_path;
      } catch (const std::exception& e) {
          LOG(ERROR) << "Failed to pre-allocate buffer " << buffer_path << ": " << e.what();
          munmap(state_, sizeof(SharedBufferPoolState));
          close(shm_fd_);
          shm_unlink(shm_name_.c_str());
          throw;
      }
    }
    
    LOG(INFO) << "BufferPool initialized in shared memory. Num buffers: " << num_buffers;
  } else {
    LOG(INFO) << "Attached to existing BufferPool in shared memory.";
  }
}

BufferPool::~BufferPool() {
  munmap(state_, sizeof(SharedBufferPoolState));
  close(shm_fd_);
  if (initialized_) {
    shm_unlink(shm_name_.c_str());
  }
}

void BufferPool::Lock() {
  pthread_mutex_lock(&state_->mutex);
}

void BufferPool::Unlock() {
  pthread_mutex_unlock(&state_->mutex);
}

std::string BufferPool::Acquire(const std::string& associated_symlink) {
  Lock();
  
  GC(); // Clean up broken symlinks

  for (size_t i = 0; i < state_->num_buffers; ++i) {
    if (state_->buffers[i].state == BufferState::kFree) {
      state_->buffers[i].state = BufferState::kAcquired;
      
      if (state_->buffers[i].object_id[0] == '\0') {
         Unlock();
         throw std::runtime_error("BufferPool: object_id is empty for free buffer!");
      }

      if (!associated_symlink.empty()) {
        snprintf(state_->buffers[i].associated_symlink, kMaxPathLen, "%s", associated_symlink.c_str());
        // Create symlink
        std::filesystem::path link_path(associated_symlink);
        std::filesystem::path target_path(state_->buffers[i].object_id);
        
        std::error_code ec;
        if (std::filesystem::exists(link_path, ec)) {
             std::filesystem::remove(link_path, ec);
        }
        std::filesystem::create_symlink(target_path, link_path, ec);
        if (ec) {
             state_->buffers[i].state = BufferState::kFree;
             state_->buffers[i].associated_symlink[0] = '\0';
             Unlock();
             throw std::runtime_error("Failed to create symlink: " + ec.message());
        }
      }

      std::string result = state_->buffers[i].object_id;
      Unlock();
      return result;
    }
  }

  Unlock();
  throw std::runtime_error("BufferPool exhausted");
}

void BufferPool::Release(const std::string& object_id) {
  Lock();
  for (size_t i = 0; i < state_->num_buffers; ++i) {
    if (object_id == state_->buffers[i].object_id) {
      state_->buffers[i].state = BufferState::kFree;
      state_->buffers[i].associated_symlink[0] = '\0';
      Unlock();
      return;
    }
  }
  Unlock();
  LOG(WARNING) << "Attempted to release unknown buffer: " << object_id;
}

void BufferPool::GC() {
  // MUST BE CALLED WITH LOCK HELD!
  for (size_t i = 0; i < state_->num_buffers; ++i) {
    if (state_->buffers[i].state == BufferState::kAcquired) {
      std::string symlink = state_->buffers[i].associated_symlink;
      if (!symlink.empty() && !std::filesystem::exists(symlink)) {
        LOG(INFO) << "GC: Releasing buffer " << state_->buffers[i].object_id << " because symlink " << symlink << " is gone.";
        state_->buffers[i].state = BufferState::kFree;
        state_->buffers[i].associated_symlink[0] = '\0';
      }
    }
  }
}

} // namespace ml_flashpoint::checkpoint_object_manager::buffer_object
