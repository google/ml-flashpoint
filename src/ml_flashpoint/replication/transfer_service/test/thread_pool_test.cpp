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

#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>

#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {
using namespace std::chrono_literals;

TEST(ThreadPoolTest, EnqueueSubmitsTaskAndReturnsCorrectValue) {
  // Given
  ThreadPool pool(4);

  // When
  auto result = pool.enqueue([](int i) { return i; }, 42);

  // Then
  EXPECT_EQ(result.get(), 42);
}

TEST(ThreadPoolTest, ConstructorDiesOnInvalidThreadCount) {
  EXPECT_DEATH(ThreadPool pool(0), "ThreadPool requires at least 1 thread.");
  EXPECT_DEATH(ThreadPool pool(-1), "ThreadPool requires at least 1 thread.");
}

TEST(ThreadPoolTest, MultipleTasksCanBeExecutedConcurrently) {
  // Given
  ThreadPool pool(4);
  std::vector<std::future<int>> results;

  // When
  for (int i = 0; i < 100; ++i) {
    results.emplace_back(pool.enqueue([i] {
      std::this_thread::sleep_for(1ms);
      return i;
    }));
  }

  // Then
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(results[i].get(), i);
  }
}

TEST(ThreadPoolTest, StopWaitsForPendingTasksAndPreventsNewTasks) {
  // Given
  ThreadPool pool(4);
  std::atomic<int> count = 0;
  for (int i = 0; i < 100; ++i) {
    pool.enqueue([&count] {
      std::this_thread::sleep_for(10ms);
      count++;
    });
  }

  // When
  pool.stop();

  // Then
  EXPECT_EQ(count, 100);
  EXPECT_THROW(pool.enqueue([] {}), std::runtime_error);
}

TEST(ThreadPoolTest, ExceptionInTaskIsPropagatedThroughFuture) {
  // Given
  ThreadPool pool(1);

  // When/Then
  auto result = pool.enqueue([] {
    throw std::runtime_error("test exception");
    return 0;
  });
  EXPECT_THROW(result.get(), std::runtime_error);
}

TEST(ThreadPoolTest, DestructorWaitsForPendingTasks) {
  // Given
  std::atomic<int> count = 0;
  {
    ThreadPool pool(4);
    for (int i = 0; i < 100; ++i) {
      pool.enqueue([&count] {
        std::this_thread::sleep_for(10ms);
        count++;
      });
    }
  }  // Destructor is called here

  // Then
  EXPECT_EQ(count, 100);
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
