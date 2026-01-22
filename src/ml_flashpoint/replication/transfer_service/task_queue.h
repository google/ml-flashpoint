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

#ifndef ML_FLASHPOINT_REPLICATION_TASK_QUEUE_H_
#define ML_FLASHPOINT_REPLICATION_TASK_QUEUE_H_

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace ml_flashpoint::replication::transfer_service {

// A thread-safe, blocking queue designed for producer-consumer scenarios.
//
// This class provides a generic, thread-safe queue for tasks of type T. It is
// particularly well-suited for situations where one or more producer threads
// are adding items to the queue, and one or more consumer threads are
// processing them.
//
// Key Features:
// - Thread Safety: All operations are internally synchronized using a
//   mutex, ensuring safe concurrent access.
// - Blocking Dequeue: The `wait_dequeue` method allows consumer threads to
//   block efficiently until a task becomes available, avoiding busy-waiting.
// - Non-Blocking Dequeue: The `try_dequeue` method provides a non-blocking
//   way to retrieve tasks, returning immediately if the queue is empty.
// - Graceful Shutdown: A `stop` mechanism is included to allow for a clean
//   shutdown of consumer threads. When the queue is stopped, waiting consumers
//   are unblocked and will stop processing once the queue is empty.
template <typename T>
class TaskQueue {
 public:
  // Enqueues an item into the queue.
  //
  // Args:
  //   item: The item to enqueue.
  void enqueue(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    condition_.notify_one();
  }

  // Dequeues an item from the queue, blocking if the queue is empty.
  //
  // This operation is thread-safe. If the queue is empty, the calling thread
  // will block until an item is enqueued or the queue is stopped. If the queue
  // is stopped and empty, `std::nullopt` is returned.
  //
  // Returns an `std::optional<T>` containing the dequeued item on success,
  // or `std::nullopt` if the queue is empty and stopped.
  std::optional<T> wait_dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !queue_.empty() || stop_; });
    if (stop_ && queue_.empty()) {
      return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  // Attempts to dequeue an item from the queue without blocking.
  //
  // This operation is thread-safe. It immediately returns an item if one is
  // available. If the queue is empty, it returns `std::nullopt` without
  // waiting.
  //
  // Returns an `std::optional<T>` containing the dequeued item on success,
  // or `std::nullopt` if the queue is empty.
  std::optional<T> try_dequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  // Returns the current number of items in the queue.
  //
  // This operation is thread-safe.
  //
  // Returns the number of items in the queue.
  size_t size() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

  // Clears all items from the queue.
  //
  // This operation is thread-safe.
  void clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    std::queue<T> empty;
    queue_.swap(empty);
  }

  // Stops the queue, signaling all waiting consumer threads to unblock.
  //
  // After this method is called, `wait_dequeue()` will return `std::nullopt`
  // once the queue is empty. New items can still be enqueued, but consumers
  // will eventually stop processing once existing items are depleted.
  void stop() {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
  }

 private:
  std::queue<T> queue_;                // Guarded by mutex_.
  std::mutex mutex_;                   // Protects queue_ and stop_.
  std::condition_variable condition_;  // Signaled when a task is added or queue
                                       // is stopped.
  bool stop_ = false;                  // Guarded by mutex_.
};

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_TASK_QUEUE_H_
