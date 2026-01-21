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

#ifndef ML_FLASHPOINT_OBJECT_MANAGER_H
#define ML_FLASHPOINT_OBJECT_MANAGER_H

#include <future>
#include <string>
#include <vector>

namespace ml_flashpoint::checkpoint_object_manager::object_manager {

/**
 * @brief Asynchronously deletes a list of directories.
 *
 * This function takes a vector of directory paths and deletes them in a
 * separate thread. It returns a future that will be ready when the deletion
 * is complete.
 *
 * @param directories A vector of strings, where each string is a path to a
 * directory to be deleted. This is a deep copy of the original list for
 * thread safety.
 *
 * @return A std::future<void> that can be used to wait for the deletion to
 * finish.
 */
std::future<void> delete_directories_async(
    std::vector<std::string> directories);

}  // namespace ml_flashpoint::checkpoint_object_manager::object_manager

#endif  // ML_FLASHPOINT_OBJECT_MANAGER_H
