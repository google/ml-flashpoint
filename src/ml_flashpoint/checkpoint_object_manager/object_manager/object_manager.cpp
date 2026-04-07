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

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <future>
#include <iostream>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_join.h"

namespace ml_flashpoint::checkpoint_object_manager::object_manager {

namespace fs = std::filesystem;

namespace {
// We use a fork/exec approach calling 'rm -rf' here instead of
// std::filesystem::remove_all to address a Segmentation Fault
// observed in multi-threaded environments. This should be safer
// and avoids the crash experienced with std::filesystem operations.
//
// The actual deletion logic
void delete_directories_task(const std::vector<std::string>& directories) {
  for (const std::string& dir_path : directories) {
    try {
      if (fs::is_directory(dir_path)) {
        LOG(INFO) << "Removing directory " << dir_path << " via fork/exec...";
        pid_t pid = fork();
        if (pid == 0) {
          // Child process
          execlp("rm", "rm", "-rf", dir_path.c_str(), (char*)NULL);
          // If execlp returns, it failed
          std::cerr << "Failed to exec rm -rf for " << dir_path << std::endl;
          exit(1);
        } else if (pid > 0) {
          // Parent process
          int status;
          waitpid(pid, &status, 0);
          if (status != 0) {
            LOG(ERROR) << "rm -rf failed for " << dir_path << " with status "
                       << status;
          }
        } else {
          LOG(ERROR) << "Failed to fork for deleting " << dir_path;
        }
      }
    } catch (const fs::filesystem_error& e) {
      // It's important to handle errors inside the thread,
      // otherwise they will cause a std::terminate.
      // For now, we'll just log to stderr.
      LOG(ERROR) << "Error deleting directory " << dir_path << ": " << e.what();
    }
  }
}
}  // namespace

std::future<void> delete_directories_async(
    std::vector<std::string> directories) {
  // 1. Create a promise to manually control the future.
  auto promise = std::make_unique<std::promise<void>>();

  // 2. Get the future, which has a _non-blocking_ destructor,
  // as it just carries the result, but does not own the thread itself.
  std::future<void> future = promise->get_future();

  if (directories.empty()) {
    promise->set_value();
    return future;
  }

  // 3. Launch a std::thread `t` for deleting the directories and updating the
  // promise.
  std::thread t([p = std::move(promise), dirs = std::move(directories)]() {
    try {
      delete_directories_task(dirs);
      p->set_value();  // Signal success
    } catch (...) {
      LOG(ERROR) << "An unexpected exception occurred when trying to delete "
                    "directories: ["
                 << absl::StrJoin(dirs, ", ") << "]";
      try {
        p->set_exception(std::current_exception());  // Signal failure
      } catch (...) {
        LOG(ERROR) << "An unexpected exception occurred when trying to set the "
                      "exception on the promise.";
      }
    }
  });

  // Detach `t` to make it a daemon thread that won't be waited on or crash when
  // it is destroyed.
  t.detach();

  return future;
}

}  // namespace ml_flashpoint::checkpoint_object_manager::object_manager
