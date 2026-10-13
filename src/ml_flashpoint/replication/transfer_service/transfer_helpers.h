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

#ifndef ML_FLASHPOINT_REPLICATION_TRANSFER_HELPERS_H_
#define ML_FLASHPOINT_REPLICATION_TRANSFER_HELPERS_H_

#include <filesystem>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"

namespace ml_flashpoint::replication::transfer_service {

std::string GenerateUuid();

// Resolves an object id to the path the service may use for it.
//
// The id is made absolute (a relative id is resolved against the process's
// current working directory, as open(2) would), symlinks in the part of it
// that exists are followed, and the rest is normalized lexically, so "..",
// "." and symlinks cannot lead outside the base directory. `base_dir` itself
// is not a valid object path.
//
// `base_dir` must be absolute and canonical, as returned by
// std::filesystem::canonical, so both sides of the comparison name the real
// location.
//
// Returns the resolved path, or InvalidArgument when the id is empty or
// resolves outside `base_dir`, or another error when the filesystem lookup
// fails (for example, a name that is too long).
absl::StatusOr<std::string> ResolvePathWithinBaseDir(
    std::string_view object_id, const std::filesystem::path& base_dir);

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_TRANSFER_HELPERS_H_
