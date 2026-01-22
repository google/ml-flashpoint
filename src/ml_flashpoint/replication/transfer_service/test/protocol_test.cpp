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

#include "protocol.h"

#include <cstring>

#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

TEST(ObjInfoHeaderTest, NullTerminateCharArraysEnsuresLastByteIsZero) {
  // Given: A header where string fields are completely filled with non-null
  // chars.
  ObjInfoHeader header;
  std::memset(header.source_obj_id, 'A', sizeof(header.source_obj_id));
  std::memset(header.dest_obj_id, 'B', sizeof(header.dest_obj_id));
  std::memset(header.source_address, 'C', sizeof(header.source_address));
  std::memset(header.dest_address, 'D', sizeof(header.dest_address));
  std::memset(header.task_id, 'E', sizeof(header.task_id));

  // When: NullTerminate is called.
  header.NullTerminateCharArrays();

  // Then: All bytes except the last one must remain unchanged, and the last
  // byte MUST be '\0'.
  for (size_t i = 0; i < sizeof(header.source_obj_id) - 1; ++i) {
    EXPECT_EQ(header.source_obj_id[i], 'A') << "at index " << i;
  }
  EXPECT_EQ(header.source_obj_id[sizeof(header.source_obj_id) - 1], '\0');

  for (size_t i = 0; i < sizeof(header.dest_obj_id) - 1; ++i) {
    EXPECT_EQ(header.dest_obj_id[i], 'B') << "at index " << i;
  }
  EXPECT_EQ(header.dest_obj_id[sizeof(header.dest_obj_id) - 1], '\0');

  for (size_t i = 0; i < sizeof(header.source_address) - 1; ++i) {
    EXPECT_EQ(header.source_address[i], 'C') << "at index " << i;
  }
  EXPECT_EQ(header.source_address[sizeof(header.source_address) - 1], '\0');

  for (size_t i = 0; i < sizeof(header.dest_address) - 1; ++i) {
    EXPECT_EQ(header.dest_address[i], 'D') << "at index " << i;
  }
  EXPECT_EQ(header.dest_address[sizeof(header.dest_address) - 1], '\0');

  for (size_t i = 0; i < sizeof(header.task_id) - 1; ++i) {
    EXPECT_EQ(header.task_id[i], 'E') << "at index " << i;
  }
  EXPECT_EQ(header.task_id[sizeof(header.task_id) - 1], '\0');

  // And: String conversions should now be safe (not read past the buffer).
  EXPECT_EQ(std::strlen(header.source_obj_id),
            sizeof(header.source_obj_id) - 1);
  EXPECT_EQ(std::strlen(header.dest_obj_id), sizeof(header.dest_obj_id) - 1);
  EXPECT_EQ(std::strlen(header.source_address),
            sizeof(header.source_address) - 1);
  EXPECT_EQ(std::strlen(header.dest_address), sizeof(header.dest_address) - 1);
  EXPECT_EQ(std::strlen(header.task_id), sizeof(header.task_id) - 1);
}

TEST(ObjInfoHeaderTest, NullTerminateDoesNotAffectAlreadyTerminatedStrings) {
  // Given: A header with short strings.
  ObjInfoHeader header;
  std::strcpy(header.dest_obj_id, "test_obj");

  // When: NullTerminate is called.
  header.NullTerminateCharArrays();

  // Then: The string content remains unchanged.
  EXPECT_STREQ(header.dest_obj_id, "test_obj");
  EXPECT_EQ(header.dest_obj_id[sizeof(header.dest_obj_id) - 1], '\0');
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
