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

#include "thread_pool.h"

#include "absl/log/check.h"

namespace ml_flashpoint::replication::transfer_service {

// Initializes the thread pool and creates the specified number of worker
// threads.
//
// Each worker thread runs a loop that waits for tasks to be added to the
// queue. When a task is available, the thread dequeues it and executes it.
// The threads will continue to run until the thread pool is stopped.
ThreadPool::ThreadPool(int threads) : stop_(false) {
  CHECK_GT(threads, 0) << "ThreadPool requires at least 1 thread.";
  for (int i = 0; i < threads; ++i) {
    workers_.emplace_back([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock<std::mutex> lock(this->queue_mutex_);
          this->condition_.wait(
              lock, [this] { return this->stop_ || !this->tasks_.empty(); });
          if (this->stop_ && this->tasks_.empty()) {
            return;
          }
          task = std::move(this->tasks_.front());
          this->tasks_.pop();
        }
        task();
      }
    });
  }
}

void ThreadPool::stop() {
  {  // Acquire lock to safely modify the stop_ flag.
    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (stop_) {
      return;
    }
    stop_ = true;
  }
  condition_.notify_all();  // Notify all workers to check the stop_ flag.
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

// Destructor for the ThreadPool.
//
// This ensures that the thread pool is stopped and all worker threads are
// joined before the object is destroyed.
ThreadPool::~ThreadPool() { stop(); }

}  // namespace ml_flashpoint::replication::transfer_service