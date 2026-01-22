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

#ifndef ML_FLASHPOINT_REPLICATION_THREAD_POOL_H_
#define ML_FLASHPOINT_REPLICATION_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ml_flashpoint::replication::transfer_service {

// Manages a pool of worker threads to execute tasks concurrently.
//
// This class provides a simple and efficient way to manage a fixed-size pool
// of threads. Tasks can be enqueued and will be executed by the next available
// worker thread. The thread pool handles the lifecycle of the threads,
// including their creation and graceful shutdown.
class ThreadPool {
 public:
  // Constructs a ThreadPool with the specified number of threads.
  //
  // Args:
  //   threads: The number of worker threads to create. Must be greater than 0.
  explicit ThreadPool(int threads);

  // Destroys the ThreadPool, stopping all worker threads and joining them.
  ~ThreadPool();

  // Enqueues a task to be executed by the thread pool.
  //
  // This method accepts a function and its arguments, wraps them in a task,
  // and adds it to the task queue. It returns a `std::future` that can be
  // used to retrieve the result of the task's execution.
  //
  // Throws a `std::runtime_error` if the thread pool has been stopped.
  //
  // Args:
  //   f: The function to execute.
  //   args: The arguments to pass to the function.
  //
  // Returns:
  //   A `std::future` representing the result of the task.
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::invoke_result_t<F, Args...>> {
    using return_type = typename std::invoke_result_t<F, Args...>;
    // Create a packaged_task to wrap the function and its arguments.
    // std::make_shared is used here because the packaged_task needs to be
    // accessible both from the future (res) and from the lambda function
    // that will be enqueued. The shared ownership ensures the packaged_task
    // remains valid until it's executed by a worker thread, even after
    // enqueue returns.
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {  // Acquire lock to safely modify the task queue.
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();  // Notify one worker that a new task is available.
    return res;
  }

  // Stops the thread pool, preventing new tasks from being enqueued and
  // waiting for all currently executing and pending tasks to complete.
  void stop();

 private:
  std::vector<std::thread> workers_;         // A vector of worker threads.
  std::queue<std::function<void()>> tasks_;  // A queue of tasks to be executed.

  std::mutex queue_mutex_;             // Mutex to protect the task queue.
  std::condition_variable condition_;  // Condition variable to signal workers.
  bool stop_;
};

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_THREAD_POOL_H_