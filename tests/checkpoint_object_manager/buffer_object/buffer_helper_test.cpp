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

#include <fcntl.h>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "absl/status/status.h"

namespace ml_flashpoint::checkpoint_object_manager::buffer_object::internal {

// This class handles the setup and teardown of a temporary file used in tests.
class BufferHelperTest : public ::testing::Test {
 protected:
  // Sets up the test environment.
  // This function is called before each test is run. It creates a unique
  // temporary path for the test buffer file.
  void SetUp() override {
    test_path_ =
        std::filesystem::temp_directory_path() / "test_buffer_object.bin";
    dir_path_ = std::filesystem::temp_directory_path() / "test_dir";
    // Ensure the test directory is created for relevant tests.
    std::filesystem::create_directory(dir_path_);
  }

  // Tears down the test environment.
  // This function is called after each test is run. It ensures all temporary
  // files and directories are deleted.
  void TearDown() override {
    // std::remove from <cstdio> is safe to call on a non-existent file.
    std::remove(test_path_.c_str());
    std::filesystem::remove_all(
        dir_path_);  // Remove the directory and its contents
  }

  // Helper function to create an empty file for testing.
  void CreateEmptyFile(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.close();
    ASSERT_TRUE(ofs) << "Test setup failed: Could not create an empty file.";
  }

  // Helper function to create a file with specific content for testing.
  void CreateFileWithContent(const std::string& path,
                             const std::string& content) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << content;
    ofs.close();
    ASSERT_TRUE(ofs) << "Failed to write content to file for test setup.";
  }

  // Helper to centralize cleanup logic for mmap and file descriptors.
  void SafeUnmapAndClose(int fd, void* ptr, size_t size) {
    if (ptr != MAP_FAILED) {
      munmap(ptr, size);
    }
    if (fd != -1) {
      close(fd);
    }
  }

  // Paths used for testing.
  std::string test_path_;
  std::string dir_path_;
};

// --- Tests for GetErrnoString and ErrnoToStatus ---

// Tests if GetErrnoString correctly converts a known error code.
TEST_F(BufferHelperTest, GetErrnoStringConvertsKnownError) {
  // ENOENT (No such file or directory) is a common and stable error code.
  std::string err_str = GetErrnoString(ENOENT);
  // We expect the output to contain "No such file or directory".
  EXPECT_THAT(err_str, ::testing::HasSubstr("No such file or directory"));
}

// Tests if ErrnoToStatus can correctly generate an absl::Status from errno.
TEST_F(BufferHelperTest, ErrnoToStatusCreatesCorrectStatus) {
  // Manually set errno.
  errno = EACCES;  // Permission denied
  absl::Status status = ErrnoToStatus("Failed to open file");

  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed to open file"));
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Permission denied"));
}

// --- Tests for create_file_and_mmap ---

// Verifies that create_file_and_mmap can successfully create and map a file
// when it does not exist.
TEST_F(BufferHelperTest, CreateAndMmapSucceedsWhenFileDoesNotExist) {
  const size_t expected_capacity = 4096;
  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  absl::Status status = create_file_and_mmap(
      test_path_, expected_capacity, fd, out_size, ptr, /*overwrite=*/false);

  ASSERT_TRUE(status.ok());
  ASSERT_NE(fd, -1);
  ASSERT_EQ(out_size, expected_capacity);
  ASSERT_NE(ptr, MAP_FAILED);

  SafeUnmapAndClose(fd, ptr, out_size);
}

// Verifies that create_file_and_mmap can successfully create and map a file
// when the directory does not exist.
TEST_F(BufferHelperTest, CreateAndMmapSucceedsWhenDirectoryDoesNotExist) {
  const size_t expected_capacity = 4096;
  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  const std::filesystem::path non_existent_dir =
      std::filesystem::temp_directory_path() / "non-exist-dir";
  const std::filesystem::path file_path = non_existent_dir / "file.bin";

  // Ensure the directory does not exist from a previous run.
  std::filesystem::remove_all(non_existent_dir);

  absl::Status status =
      create_file_and_mmap(file_path.string(), expected_capacity, fd, out_size,
                           ptr, /*overwrite=*/false);

  ASSERT_TRUE(status.ok());
  ASSERT_NE(fd, -1);
  ASSERT_EQ(out_size, expected_capacity);
  ASSERT_NE(ptr, MAP_FAILED);

  SafeUnmapAndClose(fd, ptr, out_size);
  std::filesystem::remove_all(non_existent_dir);
}

// Verifies that the function can successfully overwrite and map an empty file
// when it already exists and overwrite=true.
TEST_F(BufferHelperTest, CreateAndMmapSucceedsWithOverwriteOnEmptyFile) {
  CreateEmptyFile(test_path_);

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;
  const size_t new_capacity = 1024;

  absl::Status status = create_file_and_mmap(test_path_, new_capacity, fd,
                                             out_size, ptr, /*overwrite=*/true);

  ASSERT_TRUE(status.ok());
  EXPECT_NE(fd, -1);
  EXPECT_NE(ptr, MAP_FAILED);
  EXPECT_EQ(out_size, new_capacity);

  SafeUnmapAndClose(fd, ptr, out_size);
}

// Verifies that the function can successfully overwrite and truncate a
// non-empty file when overwrite=true.
TEST_F(BufferHelperTest, CreateAndMmapSucceedsAndTruncatesNonEmptyFile) {
  CreateFileWithContent(test_path_, "This is the initial, longer content.");

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;
  const size_t truncated_capacity = 4;

  absl::Status status = create_file_and_mmap(test_path_, truncated_capacity, fd,
                                             out_size, ptr, /*overwrite=*/true);

  ASSERT_TRUE(status.ok());
  ASSERT_NE(fd, -1);
  ASSERT_NE(ptr, MAP_FAILED);
  EXPECT_EQ(out_size, truncated_capacity);

  struct stat file_stat;
  ASSERT_EQ(stat(test_path_.c_str(), &file_stat), 0);
  EXPECT_EQ(file_stat.st_size, truncated_capacity);

  SafeUnmapAndClose(fd, ptr, out_size);
}

// Verifies that the function call fails when the file exists and
// overwrite=false.
TEST_F(BufferHelperTest, CreateAndMmapFailsWhenFileExistsAndOverwriteIsFalse) {
  const size_t expected_capacity = 1024;
  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  CreateEmptyFile(test_path_);
  absl::Status status = create_file_and_mmap(
      test_path_, expected_capacity, fd, out_size, ptr, /*overwrite=*/false);

  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kAlreadyExists);
  // EEXIST indicates that the file already exists.
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr(
                  "File already exists and overwrite is set to false."));

  ASSERT_EQ(fd, -1);
  ASSERT_EQ(ptr, MAP_FAILED);
}

// Verifies that the function call fails when the requested size is 0.
TEST_F(BufferHelperTest, CreateAndMmapFailsWithZeroSize) {
  const size_t zero_size = 0;
  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  absl::Status status = create_file_and_mmap(
      test_path_, zero_size, fd, out_size, ptr, /*overwrite=*/false);

  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

// --- Tests for open_file_and_mmap_ro ---

// Verifies that opening an existing file in READ_ONLY mode succeeds.
TEST_F(BufferHelperTest, OpenAndMmapSucceedsInReadOnlyMode) {
  const std::string content = "This is a test file.";

  CreateFileWithContent(test_path_, content);

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  absl::Status status = open_file_and_mmap_ro(test_path_, fd, out_size, ptr);

  // The operation should be successful.
  ASSERT_TRUE(status.ok());
  ASSERT_NE(fd, -1);
  ASSERT_EQ(out_size, content.size());
  ASSERT_NE(ptr, MAP_FAILED);

  // The mapped memory should contain the correct content.
  ASSERT_EQ(memcmp(ptr, content.c_str(), content.size()), 0);

  // Clean up.
  SafeUnmapAndClose(fd, ptr, out_size);
}

// Verifies that attempting to open a non-existent file fails.
TEST_F(BufferHelperTest, OpenAndMmapFailsForNonExistentFile) {
  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  // Attempt to open a path that does not exist.
  absl::Status status =
      open_file_and_mmap_ro("non_existent_file.bin", fd, out_size, ptr);

  // The operation should fail.
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
  // ENOENT indicates the file was not found.
  EXPECT_THAT(status.message(), ::testing::HasSubstr(GetErrnoString(ENOENT)));
}

// Verifies that attempting to open a directory fails.
// This version uses a dedicated directory path for clarity.
TEST_F(BufferHelperTest, OpenAndMmapFailsForDirectory) {
  // Create a directory at the dedicated directory path.
  std::filesystem::create_directory(dir_path_);

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  // Attempt to open the directory path as a file.
  absl::Status status = open_file_and_mmap_ro(dir_path_, fd, out_size, ptr);

  // The operation should fail because the path is a directory.
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  // EISDIR is the specific error code indicating the path is a directory.
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Path is a directory"));
}

// Verifies that attempting to open an empty (zero-byte) file fails.
TEST_F(BufferHelperTest, OpenAndMmapFailsForEmptyFile) {
  // Arrange: Create an empty file.
  // Note: We use CreateEmptyFile here because create_file_and_mmap does not
  // support creating zero-sized files, so we must use a different mechanism
  // to create this specific test condition.
  CreateEmptyFile(test_path_);

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;

  // Attempt to open the zero-byte file.
  absl::Status status = open_file_and_mmap_ro(test_path_, fd, out_size, ptr);

  // The operation should fail with an invalid argument error.
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("File cannot be empty"));
}

// --- Tests for unmap_and_close ---

// Verifies that unmap_and_close successfully unmaps memory and closes the file
// descriptor.
TEST_F(BufferHelperTest, UnmapAndCloseSucceedsOnValidInputs) {
  const size_t expected_capacity = 1024;

  // Manually create a file and memory map to get a valid fd and pointer.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, expected_capacity), 0);
  void* ptr =
      mmap(NULL, expected_capacity, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  absl::Status unmap_status =
      unmap_and_close(fd, ptr, expected_capacity, std::nullopt);

  ASSERT_TRUE(unmap_status.ok());

  // To confirm the fd is closed, we try to use it again.
  // A call to fstat on a closed fd should fail with EBADF (Bad file
  // descriptor).
  struct stat sb;
  errno = 0;
  ASSERT_EQ(fstat(fd, &sb), -1);
  ASSERT_EQ(errno, EBADF);
}

// Verifies that unmap_and_close correctly truncates the file to a smaller size.
TEST_F(BufferHelperTest, UnmapAndCloseWithTruncateReducesFileSize) {
  const size_t initial_size = 4096;
  const size_t truncate_to_size = 512;

  // Manually create a file with an initial size and map it to memory.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, initial_size), 0);
  void* ptr =
      mmap(NULL, initial_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);
  ASSERT_EQ(std::filesystem::file_size(test_path_), initial_size);

  // Call the function with a specific truncate size.
  absl::Status unmap_status =
      unmap_and_close(fd, ptr, initial_size, truncate_to_size);

  // The function should succeed and the file size on disk should be updated.
  ASSERT_TRUE(unmap_status.ok());
  ASSERT_EQ(std::filesystem::file_size(test_path_), truncate_to_size);
}

// Verifies that unmap_and_close returns an error if only munmap fails,
// but still attempts to close the file descriptor.
TEST_F(BufferHelperTest, UnmapAndCloseFailsOnMunmapErrorOnly) {
  const size_t test_size = 1024;

  // Create a valid file and memory map.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, test_size), 0);
  void* ptr = mmap(NULL, test_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // Call with an invalid pointer to cause munmap to fail.
  // No truncation is requested.
  void* invalid_ptr = reinterpret_cast<void*>(1);
  absl::Status status =
      unmap_and_close(fd, invalid_ptr, test_size, std::nullopt);

  // The operation should fail, reporting only the munmap error.
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), testing::HasSubstr("munmap() failed"));
  EXPECT_THAT(status.message(),
              testing::Not(testing::HasSubstr("close() failed")));

  // Verify that the file descriptor was still closed.
  errno = 0;
  ASSERT_EQ(close(fd), -1);
  ASSERT_EQ(errno, EBADF);

  // Manually unmap the original valid pointer.
  munmap(ptr, test_size);
}

// Verifies that unmap_and_close fails when a truncate request is made on a
// read-only file.
TEST_F(BufferHelperTest, UnmapAndCloseFailsOnTruncateWithReadOnlyFile) {
  const std::string initial_content =
      "This is the original content that should not be modified.";
  const size_t test_size = initial_content.size();

  // Open the file in read-write mode to set its size.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(write(fd, initial_content.c_str(), test_size), test_size);
  close(fd);
  const size_t initial_file_size = std::filesystem::file_size(test_path_);
  ASSERT_GT(initial_file_size, 0);

  // Reopen as read-only.
  fd = open(test_path_.c_str(), O_RDONLY);
  ASSERT_NE(fd, -1);
  void* ptr = mmap(NULL, test_size, PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  absl::Status status =
      unmap_and_close(fd, ptr, test_size, /*truncate_size=*/512);

  // The operation should fail because ftruncate will be called on a
  // read-only file descriptor, which is an error.
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), testing::HasSubstr("ftruncate() failed"));
  // Assert that the file size has not changed.
  EXPECT_EQ(std::filesystem::file_size(test_path_), initial_file_size);

  // Even on failure, the file descriptor should still be closed to
  // prevent resource leaks.
  struct stat sb;
  errno = 0;
  ASSERT_EQ(fstat(fd, &sb), -1);
  ASSERT_EQ(errno, EBADF);

  // Verify that the file content remains unchanged.
  std::ifstream infile{test_path_};
  ASSERT_TRUE(infile.is_open());
  std::string final_content{std::istreambuf_iterator<char>(infile),
                            std::istreambuf_iterator<char>()};
  EXPECT_EQ(final_content, initial_content);
  infile.close();
}

// Verifies that unmap_and_close returns errors if both ftruncate and close
// fail.
TEST_F(BufferHelperTest, UnmapAndCloseFailsOnFtruncateAndCloseError) {
  const size_t test_size = 1024;

  // Create a valid mapped file.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, test_size), 0);
  void* ptr = mmap(NULL, test_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // Prematurely close the fd to cause both ftruncate() and the subsequent
  // close() inside unmap_and_close() to fail.
  close(fd);

  // Call the function with the now-closed fd.
  absl::Status status =
      unmap_and_close(fd, ptr, test_size, /*truncate_size=*/512);

  // munmap should succeed, but the other two operations should fail.
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::Not(testing::HasSubstr("munmap() failed")));
  EXPECT_THAT(status.message(), testing::HasSubstr("ftruncate() failed"));
  EXPECT_THAT(status.message(), testing::HasSubstr("close() failed"));
}

// Verifies that the function correctly handles the case where fd is a sentinel.
TEST_F(BufferHelperTest, UnmapAndCloseSucceedsWithSentinelFdOnly) {
  const size_t test_size = 1024;

  // Manually create a file and memory map to get a valid pointer.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, test_size), 0);
  void* ptr = mmap(NULL, test_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // Call the function with a sentinel fd and the valid pointer.
  absl::Status status = unmap_and_close(-1, ptr, test_size, std::nullopt);

  // The operation should succeed because munmap is valid, and close is skipped.
  ASSERT_TRUE(status.ok());

  // The original fd should still be open because it was not passed to close.
  // We can confirm this by successfully closing it now.
  ASSERT_EQ(close(fd), 0);
}

// Verifies that the function correctly handles the case where data_ptr is a
// sentinel.
TEST_F(BufferHelperTest, UnmapAndCloseSucceedsWithSentinelDataPtrOnly) {
  const size_t test_size = 1024;

  // Manually create a file and memory map to get a valid fd and pointer.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, test_size), 0);
  void* ptr = mmap(NULL, test_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // Call the function with a sentinel pointer and the valid fd.
  absl::Status status = unmap_and_close(fd, MAP_FAILED, 0, std::nullopt);

  // The operation should succeed because close is valid, and munmap is skipped.
  ASSERT_TRUE(status.ok());

  // The file descriptor should have been closed. A subsequent close will fail.
  errno = 0;
  ASSERT_EQ(close(fd), -1);
  ASSERT_EQ(errno, EBADF);

  // Since the memory was not unmapped by the call, unmap it now.
  ASSERT_EQ(munmap(ptr, test_size), 0);
}

// Verifies that the function handles both arguments as sentinels, resulting in
// a no-op.
TEST_F(BufferHelperTest, UnmapAndCloseHandlesBothSentinelsAsNoOp) {
  // Call the function with both sentinel values.
  // The function should return success without performing any action.
  ASSERT_TRUE(unmap_and_close(-1, MAP_FAILED, 0, std::nullopt).ok());
}

// Verifies that the function fails and reports both errors when both inputs
// are invalid (but non-sentinel) values.
TEST_F(BufferHelperTest, UnmapAndCloseFailsWithInvalidNonSentinelInputs) {
  // Use invalid but non-sentinel values.
  int invalid_fd = 999;
  void* invalid_ptr = reinterpret_cast<void*>(1);
  size_t dummy_size = 4096;

  // Call the function with these invalid inputs.
  absl::Status status =
      unmap_and_close(invalid_fd, invalid_ptr, dummy_size, std::nullopt);

  // The operation should fail, and the error message should report
  // failures from both munmap() and close().
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), testing::HasSubstr("munmap() failed"));
  EXPECT_THAT(status.message(), testing::HasSubstr("close() failed"));
}

// Verifies that unmap_and_close fails when a truncate size is provided
TEST_F(BufferHelperTest, UnmapAndCloseFailsOnInvalidTruncateRequest) {
  // Call the function with an invalid fd (-1) but request truncation.
  absl::Status status =
      unmap_and_close(-1, MAP_FAILED, 0, /*truncate_size=*/1024);

  // The operation should fail with an internal error.
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);

  // The error message should clearly indicate it's a programming error.
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Programming error: truncate was requested, "
                                 "but the file descriptor is invalid."));
}

// that unmap_and_close fails when called with a valid pointer
// but a size of 0.
TEST_F(BufferHelperTest, UnmapAndCloseFailsWithValidPtrAndZeroSize) {
  const size_t test_size = 1024;

  // Manually create a file and map it to get a valid fd and pointer.
  int fd = open(test_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_EQ(ftruncate(fd, test_size), 0);
  void* ptr = mmap(NULL, test_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // Call the function with a valid pointer but a size of 0.
  absl::Status status = unmap_and_close(fd, ptr, 0, std::nullopt);

  // The operation should fail with an internal error.
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);

  // The error message should indicate the specific programming error.
  EXPECT_THAT(
      status.message(),
      testing::HasSubstr("Programming error: unmap_and_close called "
                         "with a valid data_ptr but a data_size of 0."));

  // --- Manual Cleanup ---
  // Since unmap_and_close failed early due to the validation error, it did not
  // perform the cleanup. We must clean up the resources here in the test.
  ASSERT_EQ(munmap(ptr, test_size), 0);

  // Verify that the file descriptor was still closed.
  errno = 0;
  ASSERT_EQ(close(fd), -1);
  ASSERT_EQ(errno, EBADF);
}

// --- Tests for resize_mmap ---

// Verifies that resizing to a larger size succeeds and preserves data.
TEST_F(BufferHelperTest, ResizeMmapSucceedsToLargerSize) {
  // Given
  const size_t initial_size = 1024;
  const size_t new_size = 2048;
  const std::string content = "Some data to preserve";

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;
  absl::Status status = create_file_and_mmap(test_path_, initial_size, fd,
                                             out_size, ptr, /*overwrite=*/true);
  ASSERT_TRUE(status.ok());

  std::memcpy(ptr, content.c_str(), content.size());

  // When
  status = resize_mmap(fd, new_size, ptr, out_size);

  // Then
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(out_size, new_size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NE(ptr, MAP_FAILED);

  // Verify content preserved
  EXPECT_EQ(std::memcmp(ptr, content.c_str(), content.size()), 0);

  // Cleanup
  SafeUnmapAndClose(fd, ptr, out_size);
}

// Verifies that resizing to a smaller size succeeds and truncates data.
TEST_F(BufferHelperTest, ResizeMmapSucceedsToSmallerSize) {
  // Given
  const size_t initial_size = 1024;
  const size_t new_size = 512;
  const std::string content = "Some data to preserve";

  int fd = -1;
  size_t out_size = 0;
  void* ptr = MAP_FAILED;
  absl::Status status = create_file_and_mmap(test_path_, initial_size, fd,
                                             out_size, ptr, /*overwrite=*/true);
  ASSERT_TRUE(status.ok());

  std::memcpy(ptr, content.c_str(), content.size());

  // When
  status = resize_mmap(fd, new_size, ptr, out_size);

  // Then
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(out_size, new_size);
  EXPECT_NE(ptr, nullptr);
  EXPECT_NE(ptr, MAP_FAILED);

  // Verify content preserved (it fits in new size)
  EXPECT_EQ(std::memcmp(ptr, content.c_str(), content.size()), 0);

  // Cleanup
  SafeUnmapAndClose(fd, ptr, out_size);
}

// Verifies that resize fails with invalid fd
TEST_F(BufferHelperTest, ResizeMmapFailsOnInvalidFd) {
  // Given
  void* ptr = nullptr;
  size_t size = 1024;

  // When
  absl::Status status = resize_mmap(-1, 2048, ptr, size);

  // Then
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

// Verifies failure when ftruncate fails (e.g., read-only fd)
TEST_F(BufferHelperTest, ResizeMmapFailsOnFtruncateFailure) {
  // Given
  CreateEmptyFile(test_path_);
  CreateFileWithContent(test_path_, "data");

  // Open as Read-Only
  int fd = open(test_path_.c_str(), O_RDONLY);
  ASSERT_NE(fd, -1);

  // Map it (read-only map)
  struct stat sb;
  fstat(fd, &sb);
  size_t size = sb.st_size;
  void* ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(ptr, MAP_FAILED);

  // When
  absl::Status status = resize_mmap(fd, size * 2, ptr, size);

  // Then
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), testing::HasSubstr("ftruncate() failed"));

  // Cleanup
  if (ptr != nullptr && ptr != MAP_FAILED) {
    munmap(ptr, size);
  }
  close(fd);
}

}  // namespace
   // ml_flashpoint::checkpoint_object_manager::buffer_object::internal
