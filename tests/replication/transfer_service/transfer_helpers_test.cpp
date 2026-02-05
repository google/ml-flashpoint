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

#include <gtest/gtest.h>

#include <regex>

namespace ml_flashpoint::replication::transfer_service {

namespace {
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
