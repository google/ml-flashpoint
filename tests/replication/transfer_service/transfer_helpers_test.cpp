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

#include "transfer_helpers.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

#include "absl/status/status.h"

namespace ml_flashpoint::replication::transfer_service {

namespace {

using ::testing::HasSubstr;

// Creates a fresh directory tree for each test:
//   <tmp>/base/            the base directory (canonical path in base_)
//   <tmp>/base/inner.txt   a regular file inside the base
//   <tmp>/outside.txt      a regular file outside the base
class ResolvePathWithinBaseDirTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("resolve_path_test_" + GenerateUuid());
    std::filesystem::create_directories(root_ / "base");
    std::ofstream(root_ / "base" / "inner.txt") << "inner";
    std::ofstream(root_ / "outside.txt") << "outside";
    root_ = std::filesystem::canonical(root_);
    base_ = root_ / "base";
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path root_;
  std::filesystem::path base_;
};

TEST_F(ResolvePathWithinBaseDirTest, AcceptsExistingFileInsideBase) {
  // Tests that a plain path inside the base resolves to itself.
  // Given
  const std::string object_id = (base_ / "inner.txt").native();

  // When
  const auto actual = ResolvePathWithinBaseDir(object_id, base_);

  // Then
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(object_id, *actual);
}

TEST_F(ResolvePathWithinBaseDirTest, AcceptsPathThatDoesNotExistYet) {
  // Tests that a destination in a directory that is not created yet is
  // accepted, because the receiving side creates files before they exist.
  // Given
  const std::string object_id = (base_ / "new_dir" / "new.obj").native();

  // When
  const auto actual = ResolvePathWithinBaseDir(object_id, base_);

  // Then
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(object_id, *actual);
}

TEST_F(ResolvePathWithinBaseDirTest, NormalizesDotSegmentsThatStayInside) {
  // Tests that "." and ".." segments are collapsed when the result is still
  // inside the base.
  // Given
  const std::string object_id =
      (base_ / "new_dir" / ".." / "." / "inner.txt").native();

  // When
  const auto actual = ResolvePathWithinBaseDir(object_id, base_);

  // Then
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ((base_ / "inner.txt").native(), *actual);
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsParentTraversal) {
  // Tests that ".." leading out of the base is rejected, whether or not the
  // escaped target exists.
  // Given
  const std::string existing = (base_ / ".." / "outside.txt").native();
  const std::string missing = (base_ / "a" / ".." / ".." / "nope").native();

  // When
  const auto actual_existing = ResolvePathWithinBaseDir(existing, base_);
  const auto actual_missing = ResolvePathWithinBaseDir(missing, base_);

  // Then
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            actual_existing.status().code());
  EXPECT_THAT(std::string(actual_existing.status().message()),
              HasSubstr("outside the base directory"));
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual_missing.status().code());
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsAbsolutePathOutsideBase) {
  // Tests that an absolute path elsewhere on the filesystem is rejected.
  // When
  const auto actual =
      ResolvePathWithinBaseDir((root_ / "outside.txt").native(), base_);

  // Then
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual.status().code());
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsSiblingWithSameNamePrefix) {
  // Tests that the check is component-wise: "<base>2/x" is not inside
  // "<base>" even though it starts with the same characters.
  // Given
  const std::filesystem::path sibling = root_ / "base2";
  std::filesystem::create_directories(sibling);
  const std::string object_id = (sibling / "x.obj").native();

  // When
  const auto actual = ResolvePathWithinBaseDir(object_id, base_);

  // Then
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual.status().code());
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsBaseDirItselfAndEmptyId) {
  // Tests that the base directory is not a valid object path and that an
  // empty id is rejected before any filesystem access.
  // When
  const auto actual_base = ResolvePathWithinBaseDir(base_.native(), base_);
  const auto actual_trailing_slash =
      ResolvePathWithinBaseDir(base_.native() + "/", base_);
  const auto actual_empty = ResolvePathWithinBaseDir("", base_);

  // Then
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual_base.status().code());
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            actual_trailing_slash.status().code());
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual_empty.status().code());
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsSymlinkInsideBasePointingOutside) {
  // Tests that a symlink placed inside the base cannot be used to reach a
  // file outside it: the link is followed before the check.
  // Given
  const std::filesystem::path link = base_ / "escape.txt";
  std::filesystem::create_symlink(root_ / "outside.txt", link);

  // When
  const auto actual = ResolvePathWithinBaseDir(link.native(), base_);

  // Then
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual.status().code());
  EXPECT_THAT(std::string(actual.status().message()),
              HasSubstr((root_ / "outside.txt").native()));
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsSymlinkedDirectoryPointingOutside) {
  // Tests that a path through a symlinked directory is resolved through the
  // link, so a directory link to the outside is rejected even when the final
  // component does not exist yet.
  // Given
  const std::filesystem::path link_dir = base_ / "dir_link";
  std::filesystem::create_directory_symlink(root_, link_dir);

  // When
  const auto actual =
      ResolvePathWithinBaseDir((link_dir / "new.obj").native(), base_);

  // Then
  EXPECT_EQ(absl::StatusCode::kInvalidArgument, actual.status().code());
}

TEST_F(ResolvePathWithinBaseDirTest, FollowsSymlinkThatStaysInside) {
  // Tests that a symlink whose target is inside the base is accepted and
  // resolved to the target.
  // Given
  const std::filesystem::path link = base_ / "alias.txt";
  std::filesystem::create_symlink(base_ / "inner.txt", link);

  // When
  const auto actual = ResolvePathWithinBaseDir(link.native(), base_);

  // Then
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ((base_ / "inner.txt").native(), *actual);
}

TEST_F(ResolvePathWithinBaseDirTest, ResolvesRelativeIdAgainstWorkingDir) {
  // Tests that a relative id is resolved against the current working
  // directory, so it is accepted when that directory is the base and rejected
  // when it is not.
  // Given
  const std::filesystem::path cwd = std::filesystem::current_path();
  const std::string object_id = "relative_id.obj";

  // When
  const auto actual_cwd_base = ResolvePathWithinBaseDir(object_id, cwd);
  const auto actual_other_base = ResolvePathWithinBaseDir(object_id, base_);

  // Then
  ASSERT_TRUE(actual_cwd_base.ok()) << actual_cwd_base.status();
  EXPECT_EQ((cwd / object_id).native(), *actual_cwd_base);
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            actual_other_base.status().code());
}

TEST_F(ResolvePathWithinBaseDirTest, RejectsOverlongComponentWithError) {
  // Tests that a filesystem lookup failure (a component longer than NAME_MAX)
  // is reported as an error instead of being accepted.
  // Given
  const std::string object_id = (base_ / std::string(300, 'a')).native();

  // When
  const auto actual = ResolvePathWithinBaseDir(object_id, base_);

  // Then
  EXPECT_FALSE(actual.ok());
}

TEST(TransferHelpersTest, GenerateUuidIsValid) {
  std::string uuid = GenerateUuid();

  // Check length
  ASSERT_EQ(uuid.length(), 36);

  // Check format with regex
  std::regex uuid_regex(
      "[0-9a-f]{8}-[0-9a-f]{4}-[4][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}");
  ASSERT_TRUE(std::regex_match(uuid, uuid_regex));

  // Check version (4)
  ASSERT_EQ(uuid[14], '4');

  // Check variant (RFC 4122)
  ASSERT_TRUE(uuid[19] == '8' || uuid[19] == '9' || uuid[19] == 'a' ||
              uuid[19] == 'b');
}

TEST(TransferHelpersTest, GenerateUuidIsUnique) {
  std::string uuid1 = GenerateUuid();
  std::string uuid2 = GenerateUuid();
  ASSERT_NE(uuid1, uuid2);
}
}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
