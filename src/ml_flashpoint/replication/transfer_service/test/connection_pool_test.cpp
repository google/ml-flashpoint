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

#include "connection_pool.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

class ConnectionPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    ASSERT_NE(listen_fd_, -1);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    ASSERT_NE(bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)), -1);
    socklen_t len = sizeof(addr);
    ASSERT_NE(getsockname(listen_fd_, (sockaddr*)&addr, &len), -1);
    port_ = ntohs(addr.sin_port);
    ASSERT_NE(listen(listen_fd_, 5), -1);
    accept_thread_ = std::thread([this]() {
      while (true) {
        int fd = accept(listen_fd_, nullptr, nullptr);
        if (fd == -1) {
          break;
        }
        close(fd);
      }
    });
  }

  void TearDown() override {
    shutdown(listen_fd_, SHUT_RDWR);
    close(listen_fd_);
    accept_thread_.join();
  }

  int port_ = 0;
  int listen_fd_ = -1;
  std::thread accept_thread_;
};

TEST_F(ConnectionPoolTest, Initialize) {
  ConnectionPool pool("127.0.0.1", port_, 5);
  EXPECT_TRUE(pool.Initialize());
}

TEST_F(ConnectionPoolTest, InitializeFailure) {
  // Use a port that is very unlikely to have a listener.
  ConnectionPool pool("127.0.0.1", port_ + 1, 1);
  EXPECT_FALSE(pool.Initialize());
}

TEST_F(ConnectionPoolTest, GetConnection) {
  ConnectionPool pool("127.0.0.1", port_, 1);
  EXPECT_TRUE(pool.Initialize());
  auto conn = pool.GetConnection();
  EXPECT_TRUE(conn.has_value());
  EXPECT_TRUE(conn->IsValid());
}

TEST_F(ConnectionPoolTest, MultipleThreads) {
  const int pool_size = 3;
  const int num_threads = 10;
  const int iterations_per_thread = 5;

  ConnectionPool pool("127.0.0.1", port_, pool_size);
  EXPECT_TRUE(pool.Initialize());

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&pool, iterations_per_thread]() {
      for (int j = 0; j < iterations_per_thread; ++j) {
        auto conn = pool.GetConnection();
        EXPECT_TRUE(conn.has_value());
        if (conn.has_value()) {
          // Simulate some work
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
          // Connection is released when conn goes out of scope
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  // After all threads are done, the pool should have all connections back.
  // We can try to get pool_size connections to verify.
  std::vector<ScopedConnection> conns;
  for (int i = 0; i < pool_size; ++i) {
    auto conn = pool.GetConnection(100);  // Short timeout
    EXPECT_TRUE(conn.has_value());
    if (conn.has_value()) {
      conns.push_back(std::move(conn.value()));
    }
  }
  // The next GetConnection should fail if the pool is full.
  EXPECT_FALSE(pool.GetConnection(100).has_value());
}

TEST_F(ConnectionPoolTest, ReleaseConnection) {
  ConnectionPool pool("127.0.0.1", port_, 1);
  EXPECT_TRUE(pool.Initialize());
  {
    auto conn = pool.GetConnection();
    EXPECT_TRUE(conn.has_value());
  }
  auto conn = pool.GetConnection();
  EXPECT_TRUE(conn.has_value());
}

TEST_F(ConnectionPoolTest, PoolExhaustion) {
  ConnectionPool pool("127.0.0.1", port_, 1);
  EXPECT_TRUE(pool.Initialize());
  auto conn1 = pool.GetConnection();
  EXPECT_TRUE(conn1.has_value());
  auto conn2 = pool.GetConnection(100);
  EXPECT_FALSE(conn2.has_value());
}

TEST_F(ConnectionPoolTest, GetConnectionInvalidTimeout) {
  ConnectionPool pool("127.0.0.1", port_, 1);
  EXPECT_TRUE(pool.Initialize());

  // Try to get a connection with a negative timeout, expecting the program to
  // terminate.
  EXPECT_DEATH(pool.GetConnection(-100), "timeout_ms must be positive");

  // Try to get a connection with a zero timeout, expecting the program to
  // terminate.
  EXPECT_DEATH(pool.GetConnection(0), "timeout_ms must be positive");
}

TEST_F(ConnectionPoolTest, ScopedConnectionMoveSemantics) {
  ConnectionPool pool("127.0.0.1", port_, 2);
  EXPECT_TRUE(pool.Initialize());

  // Test move constructor
  {
    auto conn1 = pool.GetConnection();
    EXPECT_TRUE(conn1.has_value());
    EXPECT_TRUE(conn1->IsValid());

    ScopedConnection conn2 = std::move(conn1.value());
    EXPECT_TRUE(conn2.IsValid());
    // After move, conn1 should no longer be valid.
    EXPECT_FALSE(conn1->IsValid());
  }

  // Test move assignment
  {
    auto conn3 = pool.GetConnection();
    EXPECT_TRUE(conn3.has_value());
    EXPECT_TRUE(conn3->IsValid());

    ScopedConnection conn4(-1, nullptr);  // Invalid initial state
    EXPECT_FALSE(conn4.IsValid());

    conn4 = std::move(conn3.value());
    EXPECT_TRUE(conn4.IsValid());
    // After move, conn3 should no longer be valid.
    EXPECT_FALSE(conn3->IsValid());
  }
}

TEST_F(ConnectionPoolTest, ScopedConnectionReleaseInvalidFdNoPool) {
  // Given an invalid file descriptor and no associated pool
  int invalid_fd = -1;

  // When a ScopedConnection is created with an invalid fd and no pool,
  // and Release() is called (implicitly by destructor or explicitly)
  {
    ScopedConnection conn(invalid_fd, nullptr);
    EXPECT_FALSE(conn.IsValid());
    EXPECT_EQ(conn.fd(), invalid_fd);
    EXPECT_NO_THROW(
        conn.Release());  // Should be safe to call Release on invalid state
  }  // Destructor calls Release() again, should also be safe

  // Then no crash should occur, and the state remains consistent.
  // (No external observable change for an invalid fd and null pool)
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
