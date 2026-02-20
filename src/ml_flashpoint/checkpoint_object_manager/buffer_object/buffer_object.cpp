// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "buffer_object.h"

#include <fcntl.h>     // For open flags like O_RDONLY,O_CREAT, O_RDWR
#include <sys/mman.h>  // For memory mapping (mmap, munmap)

#include "absl/log/log.h"
#include "buffer_helper.h"

// --- Constructors ---
// Constructor for creating a new buffer.
BufferObject::BufferObject(const std::string& object_id, size_t capacity,
                           bool overwrite)

    // Initialize all members to a safe, "closed" state.
    : object_id_(object_id),
      fd_(-1),
      mmap_prot_(PROT_READ | PROT_WRITE),
      capacity_(0),
      data_ptr_(nullptr),
      closed_(true) {
  // Delegate the actual work (open, truncate, mmap) to a helper function.
  //    Note: We pass the constructor's 'capacity' argument here, not the member
  //    'capacity_'.
  LOG(INFO) << "BufferObject::BufferObject: Creating object_id=" << object_id_
            << ", capacity=" << capacity << ", overwrite=" << overwrite;

  absl::Status status = ml_flashpoint::checkpoint_object_manager::
      buffer_object::internal::create_file_and_mmap(
          object_id, capacity, fd_, capacity_, data_ptr_, overwrite);

  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }

  // If successful, mark the object as open.
  closed_ = false;
}

// Constructor for opening an existing buffer in read-only mode.
BufferObject::BufferObject(const std::string& object_id)
    // Initialize all members to a safe, "closed" state.
    : object_id_(object_id),
      fd_(-1),
      mmap_prot_(PROT_READ),
      capacity_(0),
      data_ptr_(nullptr),
      closed_(true) {
  LOG(INFO) << "BufferObject::BufferObject: Opening object_id=" << object_id_
            << " with OpenMode";

  // Delegate the actual work (open, fstat, mmap) to a helper function.
  absl::Status status =
      ml_flashpoint::checkpoint_object_manager::buffer_object::internal::
          open_file_and_mmap_ro(object_id, fd_, capacity_, data_ptr_);

  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }

  // If successful, mark the object as open.
  closed_ = false;
}

BufferObject::BufferObject(BufferObject&& other) noexcept
    : object_id_(std::move(other.object_id_)),
      data_ptr_(other.data_ptr_),
      mmap_prot_(other.mmap_prot_),
      capacity_(other.capacity_),
      closed_(other.closed_),
      fd_(other.fd_) {
  other.object_id_ = "";
  other.data_ptr_ = nullptr;
  other.fd_ = -1;
  other.capacity_ = 0;
  other.closed_ = true;
}

BufferObject& BufferObject::operator=(BufferObject&& other) noexcept {
  if (this != &other) {
    close();
    object_id_ = std::move(other.object_id_);
    data_ptr_ = other.data_ptr_;
    mmap_prot_ = other.mmap_prot_;
    capacity_ = other.capacity_;
    closed_ = other.closed_;
    fd_ = other.fd_;

    other.object_id_ = "";
    other.data_ptr_ = nullptr;
    other.fd_ = -1;
    other.capacity_ = 0;
    other.closed_ = true;
  }
  return *this;
}

BufferObject::~BufferObject() noexcept {
  LOG(INFO) << "BufferObject::~BufferObject: Destroying object_id="
            << object_id_;
  close();
}

// --- Public Methods ---

std::string BufferObject::get_id() const { return object_id_; }

size_t BufferObject::get_capacity() const {
  if (is_closed()) {
    return 0;
  }
  return capacity_;
}

void* BufferObject::get_data_ptr() const { return data_ptr_; }

bool BufferObject::is_closed() const {
  return closed_ || data_ptr_ == nullptr || data_ptr_ == MAP_FAILED ||
         fd_ == -1;
}

bool BufferObject::is_readonly() const {
  // The buffer is considered read-only if the PROT_WRITE flag is NOT set
  // in its mmap protection flags.
  return !(mmap_prot_ & PROT_WRITE);
}

void BufferObject::close(std::optional<size_t> truncate_size) noexcept {
  if (is_closed()) {
    return;
  }
  LOG(INFO) << "BufferObject::close: Closing object_id=" << object_id_;

  // Truncate request on a read-only buffer is ignored with a warning.
  if (this->is_readonly() && truncate_size.has_value()) {
    LOG(WARNING)
        << "Attempting to truncate a read-only buffer. Ignoring truncate_size ("
        << truncate_size.value() << ").";
    truncate_size = std::nullopt;
  }

  // Delegate the actual unmap and close operations to a helper.
  absl::Status status =
      ml_flashpoint::checkpoint_object_manager::buffer_object::internal::
          unmap_and_close(fd_, data_ptr_, capacity_, truncate_size);

  // If the close operation fails, log an error.
  if (!status.ok()) {
    // An error during resource cleanup (e.g., in a destructor) should not
    // throw an exception, as this can lead to program termination. Throwing
    // from a destructor is unsafe and violates the noexcept specification.
    // Instead, we log the error to report the issue without crashing the
    // application. The primary goal here
    LOG(ERROR) << "Failed to close buffer object '" << object_id_
               << "': " << status;
  }

  // Reset all members to their invalid state for safety.
  closed_ = true;
  fd_ = -1;
  data_ptr_ = nullptr;
  capacity_ = 0;
}

void BufferObject::resize(size_t new_capacity) {
  if (is_closed()) {
    throw std::runtime_error("Cannot resize a closed buffer.");
  }
  if (is_readonly()) {
    throw std::runtime_error("Cannot resize a read-only buffer.");
  }
  if (new_capacity == 0) {
    throw std::runtime_error("Cannot resize buffer to 0.");
  }

  LOG(INFO) << "BufferObject::resize: Resizing object_id=" << object_id_
            << " from " << capacity_ << " to " << new_capacity;

  absl::Status status =
      ml_flashpoint::checkpoint_object_manager::buffer_object::internal::
          resize_mmap(fd_, new_capacity, data_ptr_, capacity_);

  if (!status.ok()) {
    throw std::runtime_error(status.ToString());
  }
}