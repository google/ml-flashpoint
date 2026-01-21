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
    pool_->ReleaseConnection(sockfd_, true);
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

std::optional<ScopedConnection> ConnectionPool::GetConnection(int timeout_ms) {
  CHECK_GT(timeout_ms, 0) << "timeout_ms must be positive";
  std::unique_lock<std::mutex> lock(mtx_);
  if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
        return !available_connections_.empty() || stopping_;
      })) {
    LOG(WARNING) << "ConnectionPool::GetConnection: timeout";
    return std::nullopt;
  }
  if (stopping_) {
    LOG(WARNING) << "ConnectionPool::GetConnection: stopping";
    return std::nullopt;
  }
  if (available_connections_.empty()) {
    // TODO: Handle the case when we run out of connections
    LOG(WARNING) << "ConnectionPool::GetConnection: no available connections";
    return std::nullopt;
  }
  int fd = available_connections_.front();
  available_connections_.pop();
  return ScopedConnection(fd, this);
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
      // TODO: Check if we need cleanup for the connection before return it to
      // the pool
      available_connections_.push(sockfd);
      cv_.notify_one();
    } else {
      LOG(INFO) << "ConnectionPool::ReleaseConnection: connection pool size "
                   "full, close connection";
      close(sockfd);
    }
  } else {
    LOG(INFO)
        << "ConnectionPool::ReleaseConnection: do not reuse, close connection";
    close(sockfd);
  }
}
}  // namespace ml_flashpoint::replication::transfer_service
