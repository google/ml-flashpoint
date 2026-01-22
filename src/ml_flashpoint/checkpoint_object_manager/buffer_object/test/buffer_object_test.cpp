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

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include <cstdlib>  // For malloc
#include <cstring>
#include <cstring>  // For memcpy and memcmp
#include <filesystem>
#include <fstream>

// This class handles the setup and teardown of a temporary file used in tests.
class BufferObjectTest : public ::testing::Test {
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
    std::filesystem::remove(test_path_);
    std::filesystem::remove_all(
        dir_path_);  // Remove the directory and its contents
    std::filesystem::remove(test_path_ + "_new");
  }

  // Helper function to create an empty file for testing.
  void CreateEmptyFile(const std::string& path) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.close();
  }

  // Paths used for testing.
  std::string test_path_;
  std::string dir_path_;
};

// --- Tests for the first constructor: BufferObject(path, capacity, overwrite)
// ---

// Verifies that a BufferObject is created successfully
// when the target file does not already exist and overwrite is false.
TEST_F(BufferObjectTest, CreateSucceedsWhenFileDoesNotExist) {
  const size_t expected_capacity = 8192;

  // The file does not exist, so creating it should succeed.
  ASSERT_NO_THROW({
    BufferObject buffer(test_path_, expected_capacity, false);
    ASSERT_FALSE(buffer.is_closed());
    ASSERT_EQ(buffer.get_capacity(), expected_capacity);
  });
}

// Verifies that creating a BufferObject with `overwrite = true`
// succeeds even when the file does not already exist.
TEST_F(BufferObjectTest, CreateWithOverwriteSucceedsWhenFileDoesNotExist) {
  const size_t expected_capacity = 8192;

  // The file does not exist, but O_CREAT flag ensures it's created
  // successfully.
  ASSERT_NO_THROW({
    BufferObject buffer(test_path_, expected_capacity, true);
    ASSERT_FALSE(buffer.is_closed());
    ASSERT_EQ(buffer.get_capacity(), expected_capacity);
  });
}

// Verifies that a BufferObject is created successfully
// by overwriting an existing file when `overwrite` is true.
TEST_F(BufferObjectTest, CreateWithOverwriteSucceedsWhenFileExists) {
  const size_t expected_capacity = 8192;

  // Arrange: Create a file beforehand to simulate the "existing file"
  // condition.
  CreateEmptyFile(test_path_);

  // Act & Assert: The constructor should succeed by overwriting the existing
  // file.
  ASSERT_NO_THROW({
    BufferObject buffer(test_path_, expected_capacity, true);
    ASSERT_FALSE(buffer.is_closed());
    ASSERT_EQ(buffer.get_capacity(), expected_capacity);
  });
}

// Verifies that creating a BufferObject fails
// when the file already exists and `overwrite` is false.
TEST_F(BufferObjectTest, CreateFailsWhenFileExistsAndOverwriteIsFalse) {
  // Arrange: Create a file to simulate the "existing file" condition.
  CreateEmptyFile(test_path_);

  // Act & Assert: The constructor should throw an exception because overwrite
  // is false.
  ASSERT_THROW(BufferObject(test_path_, 1024, false), std::runtime_error);
}

// Test case to verify that creating a BufferObject with zero capacity
// correctly throws an exception.
TEST_F(BufferObjectTest, CreateFailsWithZeroCapacity) {
  ASSERT_THROW(BufferObject(test_path_, 0, false), std::runtime_error);
}

// --- Tests for the second constructor: BufferObject(path) ---

// Test case to verify that attempting to open a non-existent file throws an
// exception.
TEST_F(BufferObjectTest, OpenNonExistentFile) {
  ASSERT_THROW((BufferObject(test_path_)), std::runtime_error);
}

// Test case to verify that attempting to open a zero-byte file throws an
// exception.
TEST_F(BufferObjectTest, OpenZeroByteFile) {
  CreateEmptyFile(test_path_);
  ASSERT_THROW((BufferObject(test_path_)), std::runtime_error);
}

// Test case to verify that attempting to open a directory throws an exception.
TEST_F(BufferObjectTest, OpenDirectory) {
  ASSERT_THROW((BufferObject(dir_path_)), std::runtime_error);
}

// Test case to verify that opening a read-write file in READ_ONLY mode
// succeeds.
TEST_F(BufferObjectTest, OpenReadWriteFileAsReadOnlySucceeds) {
  // Create a file with standard read-write permissions.
  // First, create a non-empty file. The "open" constructor requires a size > 0.
  {
    std::ofstream ofs(test_path_, std::ios::binary);
    ofs.write("test", 4);
  }
  // Ensure the file has read and write permissions for the user.
  chmod(test_path_.c_str(), S_IRUSR | S_IWUSR);

  // Attempt to open this file in read-only mode.
  // This should always be allowed and should not throw an exception.
  ASSERT_NO_THROW({ BufferObject reader(test_path_); });
}

// Test case to verify that opening a file in READ_WRITE mode
// allows successful writes to the buffer.
TEST_F(BufferObjectTest, OpenWithReadWriteAllowsWriting) {
  const std::string new_data = "New Data Written!";

  // Open the file in READ_WRITE mode.
  // The first constructor is used here to create the file.
  BufferObject writer(test_path_, 1024, /*overwrite=*/true);

  // Write new data directly to the buffer.
  std::memcpy(writer.get_data_ptr(), new_data.c_str(), new_data.size());

  // Verify that the data in the buffer matches what was written.
  ASSERT_EQ(
      std::memcmp(writer.get_data_ptr(), new_data.c_str(), new_data.size()), 0);

  // The destructor will handle closing and syncing to disk. We don't need to
  // reopen to verify the in-memory write was successful.
}

// Test case to verify that opening a file in READ_ONLY mode
// prevents writes to the buffer.
TEST_F(BufferObjectTest, OpenWithReadOnlyPreventsWriting) {
  // Create a file with some initial data.
  {
    BufferObject writer(test_path_, 1024, /*overwrite=*/true);
    writer.get_data_ptr();  // Just to ensure it's valid.
  }

  //
  //  Open the file in READ_ONLY mode.
  BufferObject reader(test_path_);

  // Any attempt to write to a read-only memory map should cause a crash
  // (segmentation fault). We use gtest's "death test" to verify this behavior.
  // The test runs the code inside {} in a separate process and checks if it
  // dies.
  ASSERT_DEATH(
      {
        // This write attempt should trigger a segmentation fault.
        std::memcpy(reader.get_data_ptr(), "illegal write", 13);
      },
      ".*");  // The ".*" regex matches any error message, which is fine for a
              // segfault.
}

// --- Tests for Public Methods ---

// Test case for the get_id() method.
TEST_F(BufferObjectTest, GetIdReturnsCorrectPath) {
  // Create a buffer with a known path.
  BufferObject buffer(test_path_, 1024);
  // The get_id() method should return the exact same path.
  ASSERT_EQ(buffer.get_id(), test_path_);
}

// Test case for the get_capacity() method.
TEST_F(BufferObjectTest, GetCapacityReturnsCorrectSize) {
  const size_t expected_capacity = 4096;
  // Test with the creating constructor.
  BufferObject buffer(test_path_, expected_capacity);
  ASSERT_EQ(buffer.get_capacity(), expected_capacity);

  // Test with the opening constructor.
  // The 'writer' scope ensures the file is closed before reopening.
  {
    BufferObject writer(test_path_ + "_new", 512);
  }
  BufferObject reader(test_path_ + "_new");
  // The capacity should be determined by the file size on disk.
  ASSERT_EQ(reader.get_capacity(), 512);
}

// Test case for the get_data_ptr() method.
TEST_F(BufferObjectTest, GetDataPtrReturnsValidPointer) {
  BufferObject buffer(test_path_, 1024);

  // The data pointer should be a valid memory address, not MAP_FAILED.
  ASSERT_NE(buffer.get_data_ptr(), MAP_FAILED);

  // Verify that the pointer is usable by writing and reading a value.
  int* data = static_cast<int*>(buffer.get_data_ptr());
  *data = 12345;
  ASSERT_EQ(*data, 12345);
}

// Test case for the is_readonly() method.
TEST_F(BufferObjectTest, IsReadOnlyReturnsCorrectState) {
  // A buffer created for writing should NOT be read-only.
  // This constructor creates a new file with read-write permissions.
  {
    BufferObject writer(test_path_, 1024);
    ASSERT_FALSE(writer.is_readonly());
  }

  // A buffer opened in READ_ONLY mode should be read-only.
  {
    BufferObject reader(test_path_);
    ASSERT_TRUE(reader.is_readonly());
  }
}

// Test case for the close() method and is_closed() state.
TEST_F(BufferObjectTest, CloseMarksBufferAsClosed) {
  // Create a valid buffer object.
  BufferObject buffer(test_path_, 1024);
  ASSERT_FALSE(buffer.is_closed());

  // Explicitly close the buffer.
  buffer.close();

  // The buffer should now report itself as closed.
  ASSERT_TRUE(buffer.is_closed());

  // Verify that subsequent calls to close don't cause issues.
  ASSERT_NO_THROW(buffer.close());
  ASSERT_TRUE(buffer.is_closed());
}

// Test case for closing a writable buffer with file truncation.
// This test verifies that when close() is called on a writable BufferObject
// with a specified size, the file on disk is correctly truncated.
TEST_F(BufferObjectTest, CloseWithTruncateReducesWritableFileSize) {
  const size_t initial_size = 2048;
  const size_t truncate_to_size = 128;

  // Create a file with a known size.
  BufferObject buffer(test_path_, initial_size);
  ASSERT_EQ(std::filesystem::file_size(test_path_), initial_size);
  ASSERT_FALSE(buffer.is_closed());

  // Close the buffer and request to truncate it to a smaller size.
  buffer.close(truncate_to_size);

  // The file size on disk should now match the truncated size.
  ASSERT_TRUE(buffer.is_closed());
  ASSERT_EQ(std::filesystem::file_size(test_path_), truncate_to_size);
}

// Test case for ensuring that a truncate request on a read-only file is
// ignored upon closing. This verifies that the close operation is robust
// and does not alter the file when it shouldn't.
TEST_F(BufferObjectTest, CloseWithTruncateIgnoresTruncateOnReadOnlyFile) {
  const size_t initial_size = 1024;
  const size_t truncate_to_size = 512;

  // Setup: Create a file on disk with a known initial size.
  {
    BufferObject writable_buffer(test_path_, initial_size);
  }
  ASSERT_EQ(std::filesystem::file_size(test_path_), initial_size);

  // Open the existing file in read-only mode.
  BufferObject readonly_buffer(test_path_);
  ASSERT_FALSE(readonly_buffer.is_closed());
  ASSERT_EQ(std::filesystem::file_size(test_path_), initial_size);

  // Attempt to close the read-only buffer while requesting a truncate.
  readonly_buffer.close(truncate_to_size);

  // Verify that the truncate request was ignored and the file size on
  // disk remains unchanged.
  ASSERT_TRUE(readonly_buffer.is_closed());
  ASSERT_EQ(std::filesystem::file_size(test_path_), initial_size);
}

// --- Tests for Move Semantics ---

TEST_F(BufferObjectTest, MoveConstructorTransfersOwnership) {
  // Given
  const size_t capacity = 1024;
  BufferObject original(test_path_, capacity);
  void* original_ptr = original.get_data_ptr();
  std::string original_id = original.get_id();

  // When
  // Create via move constructor
  BufferObject moved(std::move(original));

  // Then
  // The new object should have the original's resources
  EXPECT_FALSE(moved.is_closed());
  EXPECT_EQ(moved.get_data_ptr(), original_ptr);
  EXPECT_EQ(moved.get_id(), original_id);
  EXPECT_EQ(moved.get_capacity(), capacity);

  // The original object should be in a closed state with reset attributes
  EXPECT_TRUE(original.is_closed());
  EXPECT_EQ(original.get_data_ptr(), nullptr);
  EXPECT_EQ(original.get_capacity(), 0);
  EXPECT_EQ(original.get_id(), "");
}

TEST_F(BufferObjectTest, MoveAssignmentTransfersOwnership) {
  // Given
  const size_t capacity1 = 1024;
  const size_t capacity2 = 2048;
  std::string path2 = test_path_ + "_2";

  BufferObject obj1(test_path_, capacity1);
  BufferObject obj2(path2, capacity2);

  void* ptr2 = obj2.get_data_ptr();
  std::string id2 = obj2.get_id();

  // When
  // Move via assignment operator
  obj1 = std::move(obj2);

  // Then
  // obj1 should now have obj2's resources
  EXPECT_FALSE(obj1.is_closed());
  EXPECT_EQ(obj1.get_data_ptr(), ptr2);
  EXPECT_EQ(obj1.get_id(), id2);
  EXPECT_EQ(obj1.get_capacity(), capacity2);

  // obj2 should be closed with reset attributes
  EXPECT_TRUE(obj2.is_closed());
  EXPECT_EQ(obj2.get_data_ptr(), nullptr);
  EXPECT_EQ(obj2.get_capacity(), 0);
  EXPECT_EQ(obj2.get_id(), "");

  std::filesystem::remove(path2);
}

TEST_F(BufferObjectTest, MoveAssignmentToSelfIsSafe) {
  // Given
  const size_t capacity = 1024;
  BufferObject obj(test_path_, capacity);
  void* original_ptr = obj.get_data_ptr();
  std::string original_id = obj.get_id();
  bool original_readonly = obj.is_readonly();

  // When
  // Self-move assignment
  obj = std::move(obj);

  // Then
  // All properties should remain unchanged.
  EXPECT_FALSE(obj.is_closed());
  EXPECT_EQ(obj.get_data_ptr(), original_ptr);
  EXPECT_EQ(obj.get_capacity(), capacity);
  EXPECT_EQ(obj.get_id(), original_id);
  EXPECT_EQ(obj.is_readonly(), original_readonly);
}
