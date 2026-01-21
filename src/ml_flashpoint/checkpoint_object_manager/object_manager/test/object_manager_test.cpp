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

#include "object_manager.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

class ObjectManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_dir_ = fs::temp_directory_path() / "object_manager_test";
    fs::create_directories(test_dir_);
  }

  void TearDown() override { fs::remove_all(test_dir_); }

  fs::path test_dir_;
};

TEST_F(ObjectManagerTest, DeleteDirectoriesAsyncDeletesExistingDirectories) {
  // Given
  fs::path dir1 = test_dir_ / "dir1";
  fs::path dir2 = test_dir_ / "dir2";
  fs::create_directory(dir1);
  fs::create_directory(dir2);
  std::vector<std::string> dirs_to_delete = {dir1.string(), dir2.string()};

  // When
  auto future = ml_flashpoint::checkpoint_object_manager::object_manager::
      delete_directories_async(dirs_to_delete);
  future.get();  // Block until deletion is complete.

  // Then
  ASSERT_FALSE(fs::exists(dir1));
  ASSERT_FALSE(fs::exists(dir2));
}

TEST_F(ObjectManagerTest, DeleteDirectoriesAsyncHandlesEmptyVector) {
  // Given
  std::vector<std::string> dirs_to_delete;

  // When
  auto future = ml_flashpoint::checkpoint_object_manager::object_manager::
      delete_directories_async(dirs_to_delete);
  future.get();  // Block until deletion is complete.

  // Then: This should not throw or crash.
}

TEST_F(ObjectManagerTest, DeleteDirectoriesAsyncDoesNotDeleteFiles) {
  // Given
  fs::path file1 = test_dir_ / "file1.txt";
  std::ofstream ofs{file1};
  ofs << "test";
  ofs.close();
  std::vector<std::string> paths_to_delete = {file1.string()};

  // When
  auto future = ml_flashpoint::checkpoint_object_manager::object_manager::
      delete_directories_async(paths_to_delete);
  future.get();  // Block until deletion is complete.

  // Then
  ASSERT_TRUE(fs::exists(file1));
}
