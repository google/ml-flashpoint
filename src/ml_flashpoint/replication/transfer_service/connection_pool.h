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

#ifndef ML_FLASHPOINT_REPLICATION_CONNECTION_POOL_H_
#define ML_FLASHPOINT_REPLICATION_CONNECTION_POOL_H_

// This file defines a thread-safe connection pool for managing TCP connections
// to a single peer. It provides two main classes:
//
// 1. `ConnectionPool`: Manages a pool of reusable TCP connections, handling
// their
//    creation, lifecycle, and thread-safe access. This class is designed to
//    reduce the overhead of repeatedly establishing new connections.
//
// 2. `ScopedConnection`: An RAII-style wrapper for a single TCP connection
//    retrieved from the `ConnectionPool`. It ensures that the connection is
//    automatically returned to the pool when it goes out of scope, simplifying
//    connection management for the user.

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>

namespace ml_flashpoint::replication::transfer_service {

class ConnectionPool;

// An RAII wrapper for a TCP connection from the `ConnectionPool`.
//
// This class ensures that a connection is automatically returned to the pool
// when the `ScopedConnection` object goes out of scope.
class ScopedConnection {
 public:
  explicit ScopedConnection(int fd, ConnectionPool* pool);
  ScopedConnection(const ScopedConnection&) = delete;
  ScopedConnection& operator=(const ScopedConnection&) = delete;
  ScopedConnection(ScopedConnection&& other) noexcept;
  ScopedConnection& operator=(ScopedConnection&& other) noexcept;
  ~ScopedConnection();

  int fd() const { return sockfd_; }
  bool IsValid() const { return sockfd_ >= 0; }
  void Release();

 private:
  int sockfd_;
  ConnectionPool* pool_;
};

// Manages a thread-safe pool of TCP connections to a single peer.
//
// This class handles the creation, distribution, and recycling of connections,
// allowing for efficient reuse and reducing the overhead of establishing new
// connections.
class ConnectionPool {
 public:
  // Default configuration constants for ConnectionPool.
  static constexpr int kDefaultMaxConnectAttempts = 5;
  static constexpr int kDefaultConnectRetryDelayMs = 100;
  static constexpr int kDefaultGetConnectionTimeoutMs = 500;

  // Constructs a ConnectionPool.
  explicit ConnectionPool(
      std::string peer_host, int peer_port, size_t pool_size,
      int max_connect_attempts = kDefaultMaxConnectAttempts,
      int connect_retry_delay_ms = kDefaultConnectRetryDelayMs);
  ~ConnectionPool();

  // Initializes the connection pool.
  // Returns true on success, false on failure.
  bool Initialize();

  // Shuts down the connection pool gracefully.
  void Shutdown();

  // Retrieves a connection from the pool, waiting for a specified duration if
  // none are available.
  //
  // This method waits for up to `timeout_ms` for a connection to become
  // available. If the wait times out or the pool is stopped, it returns
  // `std::nullopt`.
  //
  // Returns an `std::optional<ScopedConnection>` containing the connection if
  // one is successfully retrieved, otherwise `std::nullopt`.
  std::optional<ScopedConnection> GetConnection(
      int timeout_ms = kDefaultGetConnectionTimeoutMs);

 private:
  friend class ScopedConnection;

  // Releases a connection back to the pool or closes it.
  void ReleaseConnection(int sockfd, bool reuse = true);

  // Creates a new connection to the peer.
  // Returns the socket file descriptor on a successful connection, or -1 on
  // failure.
  int CreateConnection();

  std::string peer_host_;
  int peer_port_;
  size_t max_size_;
  std::queue<int> available_connections_;  // Guarded by mtx_.
  std::mutex mtx_;  // Protects available_connections_ and stopping_.
  std::condition_variable
      cv_;  // Signaled when a connection is released or the pool is stopping.
  bool stopping_;  // Guarded by mtx_.
  int max_connect_attempts_;
  std::chrono::milliseconds connect_retry_delay_;
};

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_CONNECTION_POOL_H_