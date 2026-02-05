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

#include "task_queue.h"

#include <future>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

TEST(TaskQueueTest, SingleThreaded) {
  TaskQueue<int> queue;
  queue.enqueue(1);
  queue.enqueue(2);
  EXPECT_EQ(queue.wait_dequeue().value(), 1);
  EXPECT_EQ(queue.wait_dequeue().value(), 2);
}

TEST(TaskQueueTest, MultipleThreads) {
  TaskQueue<int> queue;
  std::thread producer([&queue]() {
    for (int i = 0; i < 100; ++i) {
      queue.enqueue(i);
    }
  });
  std::thread consumer([&queue]() {
    for (int i = 0; i < 100; ++i) {
      EXPECT_EQ(queue.wait_dequeue().value(), i);
    }
  });
  producer.join();
  consumer.join();
}

TEST(TaskQueueTest, Stop) {
  TaskQueue<int> queue;
  queue.enqueue(1);
  queue.stop();
  EXPECT_EQ(queue.wait_dequeue().value(), 1);
  EXPECT_EQ(queue.wait_dequeue(), std::nullopt);
}

TEST(TaskQueueTest, TryDequeue) {
  TaskQueue<int> queue;
  queue.enqueue(1);
  EXPECT_EQ(queue.try_dequeue().value(), 1);
  EXPECT_EQ(queue.try_dequeue(), std::nullopt);
}

TEST(TaskQueueTest, Size) {
  TaskQueue<int> queue;
  EXPECT_EQ(queue.size(), 0);
  queue.enqueue(1);
  EXPECT_EQ(queue.size(), 1);
  queue.wait_dequeue();
  EXPECT_EQ(queue.size(), 0);
}

TEST(TaskQueueTest, Clear) {
  TaskQueue<int> queue;
  queue.enqueue(1);
  queue.enqueue(2);
  queue.clear();
  EXPECT_EQ(queue.size(), 0);
  EXPECT_EQ(queue.try_dequeue(), std::nullopt);
}

TEST(TaskQueueTest, StopWithMultipleThreads) {
  TaskQueue<int> queue;
  std::vector<std::thread> consumers;
  std::atomic<int> count = 0;
  for (int i = 0; i < 4; ++i) {
    consumers.emplace_back([&queue, &count]() {
      while (auto item = queue.wait_dequeue()) {
        count++;
      }
    });
  }
  std::thread producer([&queue]() {
    for (int i = 0; i < 100; ++i) {
      queue.enqueue(i);
    }
    queue.stop();
  });
  producer.join();
  for (auto& consumer : consumers) {
    consumer.join();
  }
  EXPECT_EQ(count, 100);
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
