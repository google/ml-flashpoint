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

#include <array>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace ml_flashpoint::replication::transfer_service {

// TODO: Use /dev/urandom or boost library instead of manaully generate uuid.
std::string GenerateUuid() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  std::array<unsigned char, 16> data;
  for (auto& byte : data) {
    byte = dis(gen);
  }

  data[6] = (data[6] & 0x0F) | 0x40;
  data[8] = (data[8] & 0x3F) | 0x80;

  std::stringstream ss;
  ss << std::hex << std::setfill('0');

  int i = 0;
  for (const auto& byte : data) {
    ss << std::setw(2) << static_cast<int>(byte);
    if (i == 3 || i == 5 || i == 7 || i == 9) {
      ss << "-";
    }
    i++;
  }
  return ss.str();
}

absl::StatusOr<std::string> ResolvePathWithinBaseDir(
    std::string_view object_id, const std::filesystem::path& base_dir) {
  if (object_id.empty()) {
    return absl::InvalidArgumentError("Object id is empty");
  }
  std::error_code ec;
  // weakly_canonical leaves a relative path relative when none of its
  // components exist yet, so make it absolute first.
  const std::filesystem::path absolute =
      std::filesystem::absolute(std::filesystem::path(object_id), ec);
  if (ec) {
    return absl::InternalError(absl::StrCat("Cannot resolve object id '",
                                            object_id, "': ", ec.message()));
  }
  // Follows symlinks in the existing prefix, then normalizes the rest, so a
  // destination that does not exist yet is handled too.
  const std::filesystem::path resolved =
      std::filesystem::weakly_canonical(absolute, ec);
  if (ec) {
    return absl::InternalError(absl::StrCat("Cannot resolve object id '",
                                            object_id, "': ", ec.message()));
  }
  // Component-wise comparison: a string prefix check would accept
  // "/base2/x" for the base "/base".
  const std::filesystem::path relative = resolved.lexically_relative(base_dir);
  if (relative.empty() || relative == "." || *relative.begin() == "..") {
    return absl::InvalidArgumentError(absl::StrCat(
        "Object id '", object_id, "' resolves to '", resolved.native(),
        "', outside the base directory '", base_dir.native(), "'"));
  }
  return resolved.native();
}

}  // namespace ml_flashpoint::replication::transfer_service