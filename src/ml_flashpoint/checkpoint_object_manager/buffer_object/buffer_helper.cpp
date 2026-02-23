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

#include "buffer_helper.h"

#include <fcntl.h>     // For file open flags like O_CREAT, O_RDWR
#include <string.h>    // For strerror_r
#include <sys/mman.h>  // For memory mapping (mmap, munmap)
#include <sys/stat.h>  // For getting file status (fstat)
#include <unistd.h>    // For POSIX API calls like close() and ftruncate()

#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace ml_flashpoint::checkpoint_object_manager::buffer_object::internal {

std::string GetErrnoString(int err_num) {
  return std::system_category().message(err_num);
}

absl::Status ErrnoToStatus(const std::string& message) {
  return absl::InternalError(
      absl::StrCat(message, ": ", GetErrnoString(errno)));
}

absl::Status create_file_and_mmap(const std::string& object_id, size_t size,
                                  int& out_fd, size_t& out_data_size,
                                  void*& out_data_ptr, bool overwrite) {
  LOG(INFO) << "create_file_and_mmap: object_id=" << object_id
            << ", size=" << size << ", overwrite=" << overwrite;

  if (size <= 0) {
    return absl::InvalidArgumentError(
        "create_file_and_mmap requires a size greater than 0 for file: " +
        object_id);
  }

  // Create a new file or clear an existing one.
  //    O_CREAT: Create if it doesn't exist.
  //    O_RDWR: Open for reading and writing.
  //    O_TRUNC: Truncate to zero length if it exists.
  //    O_EXCL: "Exclusive" flag, fail if the file already exists.
  //    0666: File permissions (readable/writable by everyone).
  int open_flags;
  if (overwrite) {
    // If overwrite is true, use O_TRUNC to clear the existing file.
    open_flags = O_CREAT | O_RDWR | O_TRUNC;
  } else {
    // If overwrite is false (the default), use O_EXCL.
    // When used with O_CREAT, O_EXCL causes open() to fail if the file already
    // exists.
    open_flags = O_CREAT | O_RDWR | O_EXCL;
  }

  // Create parent directories if they don't exist.
  // TODO: Add tests for this.
  std::filesystem::path file_path(object_id);
  if (file_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(file_path.parent_path(), ec);
    if (ec) {
      return absl::InternalError(absl::StrCat(
          "Failed to create directories for: ", object_id, ": ", ec.message()));
    }
  }

  int fd = open(object_id.c_str(), open_flags, 0666);
  if (fd == -1) {
    // If the failure is due to the file already existing (EEXIST) and we are
    // in "no-overwrite" mode, throw a more specific error.
    if (errno == EEXIST && !overwrite) {
      return absl::AlreadyExistsError(
          "open() failed for file: " + object_id +
          ". File already exists and overwrite is set to false.");
    }
    // For all other errors, throw a generic open() failure error.
    return ErrnoToStatus("open() failed for file: " + object_id);
  }

  // Set the size of the file.
  if (ftruncate(fd, size) == -1) {
    close(fd);
    return ErrnoToStatus("ftruncate() failed for file: " + object_id);
  }

  // Map the file into memory.
  void* ptr = MAP_FAILED;
  ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return ErrnoToStatus("mmap() failed for file: " + object_id);
  }

  // On success, set all the output parameters.
  out_fd = fd;
  out_data_size = size;
  out_data_ptr = ptr;

  LOG(INFO) << "Successfully created and mmapped file for object '" << object_id
            << "'. data_ptr=" << out_data_ptr << ", size=" << out_data_size;

  return absl::OkStatus();
}

absl::Status open_file_and_mmap_ro(const std::string& object_id, int& out_fd,
                                   size_t& out_data_size, void*& out_data_ptr) {
  // @brief Opens an existing file in read-only mode and maps it into memory.
  //
  // This function handles opening a file, checking its status, and creating a
  // read-only memory-mapped region. It includes checks to ensure
  // the path is a file (not a directory) and that the file is not empty.
  //
  // @param object_id The path to the file to open and map.
  // @param[out] out_fd The file descriptor of the opened file.
  // @param[out] out_data_size The size of the mapped file in bytes.
  // @param[out] out_data_ptr A pointer to the beginning of the memory-mapped
  //                         file data.
  // @return absl::OkStatus() on success, or an error status if any step fails
  //         (e.g., file not found, is a directory, or is empty).

  // If the file is a directory, return an error.
  if (std::filesystem::is_directory(object_id)) {
    return absl::InvalidArgumentError("Path is a directory, not a file: " +
                                      object_id);
  }

  // Open the file with hardcoded read-only flags (`O_RDONLY`).
  int fd = open(object_id.c_str(), O_RDONLY);
  if (fd == -1) {
    return ErrnoToStatus("open() failed for file: " + object_id);
  }

  //  Get the size of the file using fstat.
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);

    return ErrnoToStatus("fstat() failed for file: " + object_id);
  }

  const size_t size = sb.st_size;

  // If the file is zero-sized, return an error.
  if (size == 0) {
    close(fd);
    return absl::InvalidArgumentError(
        "File cannot be empty, open_file_and_mmap cannot handle zero-sized "
        "file: " +
        object_id);
  }

  // Map the file into memory with hardcoded read-only protection (`PROT_READ`).
  void* ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    close(fd);
    return ErrnoToStatus("mmap() failed for file: " + object_id);
  }

  // On success, set all the output parameters.
  out_fd = fd;
  out_data_size = size;
  out_data_ptr = ptr;

  LOG(INFO) << "Successfully opened and mmapped file for object '" << object_id
            << "'. data_ptr=" << out_data_ptr << ", size=" << out_data_size;

  return absl::OkStatus();
}

absl::Status unmap_and_close(int fd, void* data_ptr, size_t data_size,
                             std::optional<size_t> truncate_size) {
  // @brief Unmaps a memory region, optionally truncates the file, and closes
  // the file descriptor.
  //
  // This function serves as a robust cleanup utility. It attempts to perform
  // all specified cleanup operations (unmap, truncate, close) even if some of
  // them fail. All errors encountered are collected and returned together in a
  // single status, ensuring that a single failure doesn't prevent other cleanup
  // steps.
  //
  // @param fd The file descriptor to close. It is safe to pass -1 if the file
  //           was not successfully opened.
  // @param data_ptr The starting address of the memory region to unmap. It is
  //                 safe to pass MAP_FAILED if the mapping failed.
  // @param data_size The size of the memory region to unmap. This should be
  //                  greater than 0 if data_ptr is valid.
  // @param truncate_size An optional new size for the file. If this value is
  //                      provided, the function will attempt to truncate the
  //                      file to this size before closing it. This parameter
  //                      should only be used for file descriptors opened in
  //                      read-write mode; providing it for a read-only file
  //                      descriptor will cause the operation to fail.
  // @return Returns absl::OkStatus() on complete success. Otherwise, returns an
  //         absl::InternalError that aggregates all error messages from the
  //         failed operations, joined by a semicolon.

  std::vector<std::string> errors;

  // --- Parameter Validation ---
  // This section checks for logically inconsistent arguments that likely
  // indicate a programming error on the caller's side. It helps catch bugs
  // early.
  if (fd == -1 && truncate_size.has_value()) {
    errors.push_back(
        "Programming error: truncate was requested, but the file descriptor is "
        "invalid.");
  }
  if (data_ptr != MAP_FAILED && data_size == 0) {
    errors.push_back(
        "Programming error: unmap_and_close called with a valid data_ptr but a "
        "data_size of 0.");
  }

  // --- Core Cleanup Logic ---
  // Unmap the memory region only if the pointer is valid and the size is
  // positive.
  if (data_ptr != MAP_FAILED && data_size > 0) {
    if (munmap(data_ptr, data_size) == -1) {
      // If munmap fails, record the specific system error.
      errors.push_back(
          absl::StrCat("munmap() failed: ", GetErrnoString(errno)));
    }
  }

  // Perform file operations only if the file descriptor is valid.
  // This avoids repeated `fd != -1` checks.
  if (fd != -1) {
    // If a truncate size is provided, attempt to truncate the file.
    if (truncate_size.has_value() &&
        ftruncate(fd, truncate_size.value()) == -1) {
      errors.push_back(
          absl::StrCat("ftruncate() failed: ", GetErrnoString(errno)));
    }
    if (close(fd) == -1) {
      errors.push_back(absl::StrCat("close() failed: ", GetErrnoString(errno)));
    }
  }

  // --- Final Error Reporting ---
  // If any errors were collected during the process, return a single status
  // containing a semicolon-separated list of all the error messages.
  if (!errors.empty()) {
    std::string error_message = absl::StrCat("Errors during unmap_and_close: ",
                                             absl::StrJoin(errors, "; "));
    LOG(ERROR) << error_message;
    return absl::InternalError(error_message);
  }

  LOG(INFO) << "Successfully unmapped memory and closed fd=" << fd
            << (truncate_size.has_value()
                    ? absl::StrCat(", truncated to size=", *truncate_size)
                    : "");
  return absl::OkStatus();
}

absl::Status resize_mmap(int fd, size_t new_size, void*& out_data_ptr,
                         size_t& out_data_size) {
  // @brief Resizes the memory map.
  //
  // This function assumes `out_data_ptr` is currently mapped with
  // `out_data_size`. It unmaps it, ftruncates the file, and remaps it with
  // `new_size`.

  if (fd == -1) {
    return absl::InvalidArgumentError("Invalid file descriptor for resize.");
  }
  if (out_data_ptr == MAP_FAILED || out_data_size == 0) {
    return absl::InvalidArgumentError(
        "Invalid data pointer or size for resize.");
  }

  // 1. Unmap existing
  if (munmap(out_data_ptr, out_data_size) == -1) {
    return ErrnoToStatus("munmap() failed during resize");
  }
  out_data_ptr = nullptr;

  // 2. Truncate
  if (ftruncate(fd, new_size) == -1) {
    return ErrnoToStatus("ftruncate() failed during resize");
  }

  // 3. Mmap new size
  void* ptr = mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    return ErrnoToStatus("mmap() failed during resize");
  }

  out_data_ptr = ptr;
  out_data_size = new_size;

  LOG(INFO) << "Successfully resized mmap to " << new_size;
  return absl::OkStatus();
}

}  // namespace
   // ml_flashpoint::checkpoint_object_manager::buffer_object::internal