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

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace ml_flashpoint::replication::transfer_service {

ScopedConnection::ScopedConnection(int fd, ConnectionPool* pool)
    : sockfd_(fd), pool_(pool) {}

ScopedConnection::ScopedConnection(ScopedConnection&& other) noexcept
    : sockfd_(other.sockfd_), pool_(other.pool_) {
  other.sockfd_ = -1;
  other.pool_ = nullptr;
}

ScopedConnection& ScopedConnection::operator=(
    ScopedConnection&& other) noexcept {
  if (this != &other) {
    Release();
    sockfd_ = other.sockfd_;
    pool_ = other.pool_;
    other.sockfd_ = -1;
    other.pool_ = nullptr;
  }
  return *this;
}

ScopedConnection::~ScopedConnection() { Release(); }

void ScopedConnection::Release() {
  if (pool_ != nullptr && sockfd_ >= 0) {
    pool_->ReleaseConnection(sockfd_, reuse_);
  } else if (sockfd_ >= 0) {
    close(sockfd_);
  }
  sockfd_ = -1;
  pool_ = nullptr;
}

ConnectionPool::ConnectionPool(std::string peer_host, int peer_port,
                               size_t pool_size, int max_connect_attempts,
                               int connect_retry_delay_ms)
    : peer_host_(std::move(peer_host)),
      peer_port_(peer_port),
      max_size_(pool_size),
      max_connect_attempts_(max_connect_attempts),
      connect_retry_delay_(connect_retry_delay_ms),
      stopping_(false) {
  CHECK_GT(max_size_, 0);
  CHECK_GT(max_connect_attempts_, 0);
  CHECK_GE(connect_retry_delay_.count(), 0);
}

ConnectionPool::~ConnectionPool() {
  {
    std::unique_lock<std::mutex> lock(mtx_);
    stopping_ = true;
  }
  cv_.notify_all();
  std::unique_lock<std::mutex> lock(mtx_);
  while (!available_connections_.empty()) {
    close(available_connections_.front());
    available_connections_.pop();
  }
}

bool ConnectionPool::Initialize() {
  std::unique_lock<std::mutex> lock(mtx_);
  if (!available_connections_.empty()) {
    return true;
  }
  stopping_ = false;
  for (size_t i = 0; i < max_size_; ++i) {
    int fd = CreateConnection();
    if (fd < 0) {
      while (!available_connections_.empty()) {
        close(available_connections_.front());
        available_connections_.pop();
      }
      return false;
    }
    available_connections_.push(fd);
  }
  return true;
}

void ConnectionPool::Shutdown() {
  std::unique_lock<std::mutex> lock(mtx_);
  stopping_ = true;
  cv_.notify_all();
}

int ConnectionPool::CreateConnection() {
  int sockfd = -1;
  for (int i = 0; i < max_connect_attempts_; ++i) {
    // If the pool is stopping, abort immediately.
    if (stopping_) {
      LOG(WARNING)
          << "ConnectionPool::CreateConnection: connection pool stopped";
      return -1;
    }

    // Step 1: Create a new socket.
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
      std::this_thread::sleep_for(connect_retry_delay_);
      continue;
    }

    // Step 2: Set up the server address structure.
    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(peer_port_);
    if (inet_pton(AF_INET, peer_host_.c_str(), &serv_addr.sin_addr) <= 0) {
      LOG(WARNING) << "ConnectionPool::CreateConnection: invalid address";
      close(sockfd);
      return -1;
    }

    // Step 3: Attempt to connect to the peer.
    if (connect(sockfd, reinterpret_cast<sockaddr*>(&serv_addr),
                sizeof(serv_addr)) == 0) {
      return sockfd;
    }

    // If the connection fails, close the socket and wait before retrying.
    close(sockfd);
    std::this_thread::sleep_for(connect_retry_delay_);
  }
  LOG(WARNING)
      << "ConnectionPool::CreateConnection: max connect attempts reached";
  return -1;
}

bool ConnectionPool::IsConnectionAlive(int sockfd) {
  if (sockfd < 0) return false;

  char buf;
  // We use MSG_PEEK | MSG_DONTWAIT to check the status of the TCP connection
  // without actually consuming any data from the socket's receive buffer.
  // This is a fast, zero-copy way to ask the kernel if the connection has
  // been closed or has encountered an error while it was idle in the pool.
  ssize_t r = recv(sockfd, &buf, 1, MSG_PEEK | MSG_DONTWAIT);

  if (r == 0) {
    // If recv returns 0, it means the remote peer has performed an orderly
    // shutdown (sent a FIN packet). The connection is no longer usable for
    // sending new requests.
    return false;
  } else if (r < 0) {
    // If recv returns -1, we check errno to distinguish between "no data"
    // and a real network error.
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      // EAGAIN/EWOULDBLOCK means the connection is still alive and healthy,
      // but there is currently no data waiting to be read. This is the
      // expected state for an idle connection in the pool.
      return true;
    }
    // Any other error (like ECONNRESET or EPIPE) indicates that the
    // connection has been broken or timed out.
    return false;
  }
  // If r > 0, there is actual data waiting in the buffer. While unusual for
  // an idle pool connection, it indicates the connection is definitely alive.
  return true;
}

std::optional<ScopedConnection> ConnectionPool::GetConnection(int timeout_ms) {
  CHECK_GT(timeout_ms, 0) << "timeout_ms must be positive";

  // Calculate the absolute deadline to ensure we respect the user-provided
  // timeout even if we have to loop through several dead connections.
  auto start_time = std::chrono::steady_clock::now();
  auto end_time = start_time + std::chrono::milliseconds(timeout_ms);

  while (true) {
    std::unique_lock<std::mutex> lock(mtx_);

    // Re-calculate the remaining wait time for each iteration of the loop.
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - std::chrono::steady_clock::now());

    if (remaining.count() <= 0 || !cv_.wait_for(lock, remaining, [this] {
          return !available_connections_.empty() || stopping_;
        })) {
      LOG(WARNING) << "ConnectionPool::GetConnection: timeout reached while "
                      "searching for a healthy connection";
      return std::nullopt;
    }

    if (stopping_) {
      LOG(WARNING) << "ConnectionPool::GetConnection: pool is shutting down";
      return std::nullopt;
    }

    // Pop the oldest connection from the FIFO queue.
    int fd = available_connections_.front();
    available_connections_.pop();

    // Verify the connection's health before handing it to the caller.
    // This protects against "stale" connections that were closed by the
    // peer or a firewall while sitting idle in the pool.
    if (IsConnectionAlive(fd)) {
      return ScopedConnection(fd, this);
    }

    // The connection is dead. We close it and attempt to retrieve another
    // one from the queue.
    LOG(INFO) << "ConnectionPool::GetConnection: discarded dead connection; "
                 "retrying with next available connection";
    close(fd);

    // To maintain the desired pool size, we immediately attempt to open a
    // replacement connection. This ensures the pool doesn't slowly drain
    // if many connections go stale at once.
    int new_fd = CreateConnection();
    if (new_fd >= 0) {
      available_connections_.push(new_fd);
      // The loop will continue and pick up this or another connection.
    }
  }
}

// Returns a connection to the pool, allowing it to be reused.
//
// If `reuse` is true and the pool is not full, the connection is added back to
// the queue of available connections. Otherwise, the connection is closed.
void ConnectionPool::ReleaseConnection(int sockfd, bool reuse) {
  if (sockfd < 0) {
    LOG(WARNING) << "ConnectionPool::ReleaseConnection: invalid sockfd";
    return;
  }

  std::unique_lock<std::mutex> lock(mtx_);
  if (stopping_) {
    LOG(WARNING)
        << "ConnectionPool::ReleaseConnection: stopping, close connection";
    close(sockfd);
    return;
  }

  if (reuse) {
    if (available_connections_.size() < max_size_) {
      LOG(INFO) << "ConnectionPool::ReleaseConnection: reuse connection";
      available_connections_.push(sockfd);
      cv_.notify_one();
    } else {
      LOG(INFO) << "ConnectionPool::ReleaseConnection: connection pool size "
                   "full, close connection";
      close(sockfd);
    }
  } else {
    LOG(INFO)
        << "ConnectionPool::ReleaseConnection: connection marked as unusable, "
           "closing and replenishing pool";
    close(sockfd);

    // Since we are discarding a connection that was previously part of the
    // pool's "active" set, we create a new one to maintain the fixed pool size.
    // This prevents the pool from permanently shrinking when network errors
    // occur.
    int new_fd = CreateConnection();
    if (new_fd >= 0) {
      available_connections_.push(new_fd);
      cv_.notify_one();
    } else {
      LOG(ERROR) << "ConnectionPool::ReleaseConnection: failed to replenish "
                    "pool after discarding unusable connection";
    }
  }
}
}  // namespace ml_flashpoint::replication::transfer_service
