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

#include "transfer_service.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "buffer_object.h"
#include "net_util.h"
#include "protocol.h"
#include "task.h"
#include "transfer_helpers.h"

namespace ml_flashpoint::replication::transfer_service {

constexpr int kMaxEvents = 128;
constexpr int kEpollTimeoutMs = 500;
constexpr mode_t kDefaultFilePermissions = 0644;

constexpr absl::Duration kThreadJoinTimeout = absl::Seconds(1);
constexpr std::string_view kTempFileSuffix = ".tmp";

static std::once_flag init_flag;

TransferService::TransferService(
    const std::optional<std::string>& local_ip_address)
    : local_ip_address_(local_ip_address) {
  // Make sure absl logging is initialized and set to kFatal.
  // Avoid duplicated logging.
  std::call_once(init_flag, []() {
    absl::InitializeLog();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);
  });

  // Create mlf_log_sink_ with default rank -1
  mlf_log_sink_ = std::make_unique<MLFLogSink>(-1);
  absl::AddLogSink(mlf_log_sink_.get());
}

TransferService::~TransferService() {
  LOG(INFO) << "TransferService::~TransferService";
  Shutdown();
  absl::RemoveLogSink(mlf_log_sink_.get());
}

int TransferService::Initialize(int listen_port, int threads,
                                int conn_pool_size_per_peer, int global_rank) {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (running_.load()) {
    LOG(INFO) << "Transfer service is already running.";
    return listen_port_;
  }
  // Validates the input parameters.
  CHECK_GT(threads, 0) << "threads must be positive";
  CHECK_GE(listen_port, 0) << "listen_port must be non-negative";
  CHECK_GT(conn_pool_size_per_peer, 0) << "conn_pool_per_peer must be positive";
  threads_ = threads;
  conn_pool_size_per_peer_ = conn_pool_size_per_peer;
  global_rank_ = global_rank;
  absl::RemoveLogSink(mlf_log_sink_.get());
  mlf_log_sink_ = std::make_unique<MLFLogSink>(global_rank);
  absl::AddLogSink(mlf_log_sink_.get());

  // Get listener fd and bind to listen port
  absl::StatusOr<int> listener_fd_or = SetupListeningSocket(listen_port);
  if (!listener_fd_or.ok()) {
    LOG(ERROR) << "Failed to setup listening socket: "
               << listener_fd_or.status();
    return -1;
  }
  listener_fd_ = *listener_fd_or;

  // Get local address
  absl::StatusOr<std::string> local_ip_address;
  if (local_ip_address_.has_value()) {
    local_ip_address = local_ip_address_.value();
  } else {
    local_ip_address = GetLocalIpAddress();
  }
  if (!local_ip_address.ok()) {
    LOG(ERROR) << "Failed to get local IP address: "
               << local_ip_address.status();
    close(listener_fd_);
    return -1;
  }
  local_address_ =
      absl::StrFormat("%s:%d", local_ip_address.value(), listen_port_);
  LOG(INFO) << "Listening on " << local_address_;

  // Create the thread pool.
  thread_pool_ = std::make_unique<ThreadPool>(threads_);

  // Create epoll instance.
  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ == -1) {
    PLOG(ERROR) << "Failed to create epoll instance";
    close(listener_fd_);
    return -1;
  }

  struct epoll_event event;
  event.events = EPOLLIN | EPOLLET;
  event.data.fd = listener_fd_;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listener_fd_, &event) == -1) {
    PLOG(ERROR) << "Failed to add listener fd to epoll";
    close(listener_fd_);
    close(epoll_fd_);
    return -1;
  }
  running_.store(true);
  task_queue_thread_ =
      std::thread(&TransferService::ProcessTaskQueueLoop, this);
  epoll_thread_ = std::thread(&TransferService::ProcessEpollEventsLoop, this);

  LOG(INFO) << "TransferService initialized and listening on port "
            << listen_port_ << " with threads=" << threads_
            << ", conn_pool_per_peer=" << conn_pool_size_per_peer_
            << ", global_rank=" << global_rank_;
  return listen_port_;
}

void TransferService::Shutdown() {
  LOG(INFO) << "TransferService::Shutdown";
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (!running_.load()) {
    return;
  }

  // 1. Signal all loops to stop.
  running_.store(false);

  // 2. Stop the task queue. This will unblock the task_queue_thread_ if it's
  // waiting for tasks.
  task_queue_.stop();

  // 3. Join the task queue thread. This ensures no new tasks will be enqueued
  // to the thread pool from the task queue.
  // TODO: Add timeout from thread join.
  if (task_queue_thread_.joinable()) {
    task_queue_thread_.join();
  }

  // Set exceptions for all unfinished tasks' futures.
  {
    std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
    for (auto const& [task_id, context] : pending_tasks_) {
      try {
        context.promise->set_exception(std::make_exception_ptr(
            std::runtime_error("Service is shutting down")));
      } catch (const std::future_error& e) {
        if (e.code() != std::future_errc::promise_already_satisfied) {
          LOG(ERROR) << "Caught exception while setting promise exception: "
                     << e.what();
        }
      }
    }
    pending_tasks_.clear();
  }

  // 4. Unblock the epoll thread by closing the listener fd. This will cause
  // epoll_wait to return.
  if (listener_fd_ != -1) {
    if (shutdown(listener_fd_, SHUT_RDWR) == -1) {
      PLOG(WARNING) << "Failed to shutdown listener_fd";
    }
    if (close(listener_fd_) == -1) {
      PLOG(WARNING) << "Failed to close listener_fd";
    }
    listener_fd_ = -1;
  }

  // 5. Join the epoll thread. This ensures no new tasks will be enqueued to the
  // thread pool from the epoll loop.
  if (epoll_thread_.joinable()) {
    epoll_thread_.join();
  }

  // 6. Stop the thread pool. This will wait for all currently executing tasks
  // to complete. Now that both the task queue and epoll threads are stopped,
  // no new tasks can be enqueued.
  if (thread_pool_) {
    thread_pool_->stop();
  }

  // 7. Clean up connection pools.
  {
    std::unique_lock write_lock(connection_pools_mutex_);
    for (auto const& [peer_addr, pool] : connection_pools_) {
      if (pool) {
        pool->Shutdown();
      }
    }
    connection_pools_.clear();
  }

  // 8. Clean up epoll fd.
  if (epoll_fd_ != -1) {
    if (close(epoll_fd_) == -1) {
      PLOG(WARNING) << "Failed to close epoll_fd";
    }
    epoll_fd_ = -1;
  }
}

absl::StatusOr<int> TransferService::SetupListeningSocket(int listen_port) {
  // Binds the socket to any available address and starts the server.
  int listener_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (listener_fd == -1) {
    return absl::ErrnoToStatus(errno, "Failed to create socket");
  }

  // Allow address reuse.
  int opt = 1;
  if (setsockopt(listener_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) ==
      -1) {
    close(listener_fd);
    return absl::ErrnoToStatus(errno, "Failed to set SO_REUSEADDR");
  }

  // Set non-blocking.
  int flags = fcntl(listener_fd, F_GETFL, 0);
  if (flags == -1 || fcntl(listener_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
    close(listener_fd);
    return absl::ErrnoToStatus(errno, "Failed to set non-blocking");
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port =
      htons(listen_port);  // Port 0 lets the OS pick an ephemeral port.

  // Bind the socket to the address.
  if (bind(listener_fd, reinterpret_cast<struct sockaddr*>(&addr),
           sizeof(addr)) == -1) {
    close(listener_fd);
    return absl::ErrnoToStatus(errno, "Failed to bind socket");
  }

  // Retrieve the assigned port.
  socklen_t addr_len = sizeof(addr);
  if (getsockname(listener_fd, reinterpret_cast<struct sockaddr*>(&addr),
                  &addr_len) == -1) {
    close(listener_fd);
    return absl::ErrnoToStatus(errno, "Failed to get socket name");
  }
  listen_port_ = ntohs(addr.sin_port);

  // Start listening.
  if (listen(listener_fd, SOMAXCONN) == -1) {
    close(listener_fd);
    return absl::ErrnoToStatus(errno, "Failed to listen on socket");
  }
  return listener_fd;
}

std::future<TransferResult> TransferService::AsyncPut(
    void* data_ptr, size_t data_size, const std::string& dest_address,
    const std::string& dest_obj_id) {
  LOG(INFO) << "TransferService::AsyncPut: data_ptr=" << data_ptr
            << ", data_size=" << data_size << ", dest_address=" << dest_address
            << ", dest_obj_id=" << dest_obj_id;
  if (!running_.load()) {
    LOG(INFO) << "Transfer service is not running.";
    std::promise<TransferResult> promise;
    promise.set_exception(
        std::make_exception_ptr(std::runtime_error("Service is not running")));
    return promise.get_future();
  }
  std::string task_id = GenerateUuid();
  auto metric_container = std::make_shared<PutTaskMetricContainer>();
  metric_container->task_id = task_id;
  metric_container->task_type = TaskMetricContainer::TaskType::kPut;
  metric_container->data_size = data_size;
  metric_container->submit_time = absl::Now();

  auto task =
      std::make_unique<PutTask>(task_id, data_ptr, data_size, dest_obj_id,
                                dest_address, metric_container);

  auto promise = std::make_shared<std::promise<TransferResult>>();
  std::future<TransferResult> future = promise->get_future();

  {
    std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
    pending_tasks_[task_id] = {promise, metric_container};
  }

  if (data_ptr == nullptr || data_size == 0) {
    ReportResult(task_id, false, "Empty buffer or null data_ptr");
    return future;
  }

  task_queue_.enqueue(std::move(task));

  return future;
}

std::future<TransferResult> TransferService::AsyncGet(
    const std::string& source_obj_id, const std::string& source_address,
    const std::string& dest_obj_id) {
  LOG(INFO) << "TransferService::AsyncGet: source_obj_id=" << source_obj_id
            << ", source_address=" << source_address
            << ", dest_obj_id=" << dest_obj_id;
  if (!running_.load()) {
    LOG(INFO) << "Transfer service is not running.";
    std::promise<TransferResult> promise;
    promise.set_exception(
        std::make_exception_ptr(std::runtime_error("Service is not running")));
    return promise.get_future();
  }
  std::string task_id = GenerateUuid();
  auto metric_container = std::make_shared<GetTaskMetricContainer>();
  metric_container->task_id = task_id;
  metric_container->task_type = TaskMetricContainer::TaskType::kGet;
  metric_container->submit_time = absl::Now();

  auto task = std::make_unique<GetTask>(task_id, source_obj_id, dest_obj_id,
                                        source_address, local_address_,
                                        metric_container);

  auto promise = std::make_shared<std::promise<TransferResult>>();
  std::future<TransferResult> future = promise->get_future();

  {
    std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
    pending_tasks_[task_id] = {promise, metric_container};
  }

  task_queue_.enqueue(std::move(task));

  return future;
}

void TransferService::ProcessTaskQueueLoop() {
  LOG(INFO) << "Task queue processing loop started.";
  while (running_.load()) {
    std::optional<TaskUniquePtr> task_opt = task_queue_.wait_dequeue();

    if (!task_opt.has_value()) {
      if (running_.load()) {
        LOG(WARNING) << "Task Queue Loop: wait_dequeue returned empty "
                        "unexpectedly.";
      } else {
        LOG(INFO) << "Task Queue Loop: Stop signal received, exiting.";
      }
      break;
    }

    // Enqueue the task execution. The lambda captures the task pointer and
    // calls its Execute method on a worker thread.
    thread_pool_->enqueue(
        [this, task = std::move(task_opt.value())]() { task->Execute(this); });
  }
  LOG(INFO) << "Task queue processing loop finished.";
}

void TransferService::ProcessEpollEventsLoop() {
  LOG(INFO) << "Epoll processing loop started.";
  struct epoll_event events[kMaxEvents];

  while (running_.load()) {
    int n_events = epoll_wait(epoll_fd_, events, kMaxEvents, kEpollTimeoutMs);
    if (n_events < 0) {
      if (errno == EINTR) continue;
      if (!running_.load() && errno == EBADF) {
        LOG(INFO) << "Epoll loop: epoll_fd closed, likely due to shutdown.";
      } else {
        PLOG(ERROR) << "epoll_wait failed";
        if (running_.load()) {
          // If running, continue to next iteration to re-evaluate state.
          continue;
        }
      }
      break;
    }

    if (!running_.load()) {
      LOG(INFO) << "Epoll loop: shutting down.";
      break;
    }

    for (int i = 0; i < n_events; ++i) {
      int current_fd = events[i].data.fd;
      uint32_t current_events = events[i].events;

      if (current_fd == listener_fd_) {
        if (current_events & EPOLLERR) {
          PLOG(ERROR) << "Epoll error on listener socket fd=" << listener_fd_
                      << ". Shutting down.";
          // If there's an error on the listener, it's unrecoverable for this
          // loop.
          running_.store(false);
          break;
        }
        thread_pool_->enqueue([this]() { this->HandleNewConnection(); });
      } else {
        if ((current_events & EPOLLERR) || (current_events & EPOLLHUP)) {
          if (current_events & EPOLLERR) {
            LOG(ERROR) << "Epoll error event (EPOLLERR) on client fd="
                       << current_fd << ". Closing.";
          }
          if (current_events & EPOLLHUP) {
            LOG(INFO) << "Epoll HUP event (EPOLLHUP) on client fd="
                      << current_fd << ". Closing.";
          }
          thread_pool_->enqueue(
              [this, current_fd]() { this->RemoveClient(current_fd); });
          continue;
        }

        if (current_events & EPOLLIN) {
          LOG(INFO) << "Epoll loop: Dispatching incoming data "
                       "processing for fd="
                    << current_fd;
          thread_pool_->enqueue(&TransferService::ProcessIncomingData, this,
                                current_fd);
        } else {
          LOG(WARNING) << "Epoll loop: Unhandled event flags=" << current_events
                       << " on client fd=" << current_fd;
        }
      }
    }
    if (!running_.load()) {
      // Break outer loop if running_ was set to false inside the event loop.
      break;
    }
  }
  LOG(INFO) << "Epoll processing loop finished.";
}

void TransferService::ReportResult(std::string task_id, bool success,
                                   const std::string& message) {
  std::shared_ptr<std::promise<TransferResult>> promise;
  std::string timing_message = "";
  std::string task_type = "";
  size_t data_size = 0;

  {
    std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
    auto it = pending_tasks_.find(task_id);
    if (it != pending_tasks_.end()) {
      promise = it->second.promise;
      timing_message = it->second.metric_container->ToString();
      task_type = TaskTypeToString(it->second.metric_container->task_type);
      data_size = it->second.metric_container->data_size;
      pending_tasks_.erase(it);
    } else {
      LOG(WARNING) << "No promise found for task_id: " << task_id;
      return;
    }
  }

  LOG(INFO) << "TransferService::report_result: task_id=" << task_id
            << ", task_type=" << task_type << ", data_size=" << data_size
            << ", success=" << success << ", message=" << message
            << ", timing=" << timing_message;

  if (promise) {
    TransferResult result;
    result.task_id = task_id;
    result.success = success;
    if (!success) {
      result.error_message = message;
    }

    if (success) {
      promise->set_value(result);
    } else {
      try {
        std::runtime_error ex(message);
        promise->set_exception(std::make_exception_ptr(ex));
      } catch (const std::exception& e) {
        LOG(ERROR) << "Caught exception while setting promise exception: "
                   << e.what();
      }
    }
  }
}

void TransferService::HandleNewConnection() {
  LOG(INFO) << "Handling new connection...";
  while (running_.load()) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int conn_fd =
        accept(listener_fd_, reinterpret_cast<struct sockaddr*>(&client_addr),
               &client_len);

    if (conn_fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No more pending connections.
        break;
      } else if (errno == EINTR) {
        // Interrupted by a signal, try again.
        LOG(WARNING) << "Accept interrupted, trying again.";
        continue;
      } else {
        // A real error occurred.
        if (running_.load()) {
          LOG(ERROR) << "Failed to accept new connection";
        }
        break;
      }
    }

    // Set the new socket to non-blocking.
    int flags = fcntl(conn_fd, F_GETFL, 0);
    if (flags == -1 || fcntl(conn_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
      PLOG(ERROR) << "Failed to set O_NONBLOCK on client socket";
      close(conn_fd);
      continue;
    }

    // Add the new client to epoll.
    struct epoll_event event;
    event.events =
        EPOLLIN | EPOLLET | EPOLLONESHOT;  // Edge-triggered for new data.
    event.data.fd = conn_fd;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, conn_fd, &event) == -1) {
      PLOG(ERROR) << "Failed to add client fd to epoll";
      close(conn_fd);
      continue;
    }

    char client_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip_str, INET_ADDRSTRLEN);
    LOG(INFO) << "Accepted new connection from " << client_ip_str
              << " on fd=" << conn_fd;
  }
}

void TransferService::RemoveClient(int client_fd) {
  if (client_fd < 0) return;
  if (epoll_fd_ >= 0) {
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
  }
  close(client_fd);
}

void TransferService::ProcessIncomingData(int client_fd) {
  LOG(INFO) << "Processing incoming data for fd=" << client_fd;
  ObjInfoHeader header;
  std::memset(&header, 0, sizeof(ObjInfoHeader));

  if (!RecvHeader(client_fd, header).ok()) {
    LOG(ERROR) << "Worker fd=" << client_fd
               << ": Failed to receive complete header or connection closed.";
    RemoveClient(client_fd);
    return;
  }

  LOG(INFO) << "Worker fd=" << client_fd
            << ": Received header type=" << static_cast<int>(header.type)
            << ", dest_obj_id=" << header.dest_obj_id;

  bool keep_connection = true;
  switch (header.type) {
    case MessageType::kPutObj:
      HandleDataReceive(client_fd, header, false);
      break;
    case MessageType::kRespondToGetObj:
      HandleDataReceive(client_fd, header, true);
      break;
    case MessageType::kGetObj:
      HandleGetObjRequest(client_fd, header);
      break;
    case MessageType::kAck:
      LOG(ERROR) << "Worker fd=" << client_fd
                 << ": Unexpected kAck message received for task "
                 << header.task_id;
      break;
    case MessageType::kError:
      LOG(ERROR) << "Worker fd=" << client_fd
                 << ": Error message received for task " << header.task_id;
      ReportResult(header.task_id, false, "Received error message");
      break;
  }

  if (!keep_connection) {
    RemoveClient(client_fd);
    LOG(INFO) << "Worker fd=" << client_fd
              << ": Finished processing, connection closed.";
  } else {
    LOG(INFO) << "Worker fd=" << client_fd
              << ": Finished processing, re-arming epoll.";
    struct epoll_event event;
    event.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
    event.data.fd = client_fd;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_MOD, client_fd, &event) < 0) {
      PLOG(ERROR) << "epoll_ctl re-arm failed";
      RemoveClient(client_fd);
    }
  }
}

void TransferService::ExecutePutTask(PutTask* task) {
  auto& metric_container = task->GetMetricContainer();
  metric_container.start_execution_time = absl::Now();
  LOG(INFO) << "Executing PutTask for task_id=" << task->GetTaskId();
  auto conn_opt = GetConnectionFromPool(task->GetDestAddr());
  if (!conn_opt) {
    ReportResult(task->GetTaskId(), false, "Failed to get connection");
    return;
  }
  ScopedConnection conn = std::move(conn_opt.value());
  metric_container.connection_acquired_time = absl::Now();

  ObjInfoHeader header;
  std::memset(&header, 0, sizeof(ObjInfoHeader));
  snprintf(header.dest_obj_id, sizeof(header.dest_obj_id), "%s",
           task->GetDestObjId().c_str());
  header.type = MessageType::kPutObj;
  header.obj_size = task->GetDataSize();

  int sockfd = conn.fd();

  if (!SendAll(sockfd, &header, kHeaderSize).ok()) {
    LOG(ERROR) << "perform_send_obj: Failed sending header/filename for "
               << task->GetDestAddr();
    ReportResult(task->GetTaskId(), false, "Failed to send header");
    return;
  }
  metric_container.header_sent_time = absl::Now();

  if (!SendAll(sockfd, task->GetDataPtr(), task->GetDataSize()).ok()) {
    LOG(ERROR) << "perform_send_obj: Failed sending data for "
               << task->GetDestAddr();
    ReportResult(task->GetTaskId(), false, "Failed to send data");
    return;
  }
  metric_container.data_sent_time = absl::Now();

  ObjInfoHeader ack_header;
  if (!RecvHeader(sockfd, ack_header).ok()) {
    ReportResult(task->GetTaskId(), false, "Failed to receive ACK");
    return;
  }
  switch (ack_header.type) {
    case MessageType::kAck:
      LOG(INFO) << "Buffer data sent and ACK received successfully !!!";
      metric_container.finish_time = absl::Now();
      ReportResult(task->GetTaskId(), true,
                   "Buffer data sent and ACK received");
      break;
    case MessageType::kError:
      ReportResult(task->GetTaskId(), false, "Received error from destination");
      break;
    case MessageType::kPutObj:
    case MessageType::kGetObj:
    case MessageType::kRespondToGetObj:
      ReportResult(task->GetTaskId(), false, "Received unexpected ACK");
      break;
  }
}

void TransferService::SendErrorResponse(int client_fd, const char* task_id,
                                        const char* dest_obj_id) {
  ObjInfoHeader err_header;
  err_header.type = MessageType::kError;
  snprintf(err_header.task_id, sizeof(err_header.task_id), "%s", task_id);
  if (!SendAll(client_fd, &err_header, kHeaderSize).ok()) {
    LOG(ERROR) << "Failed to send error response for " << dest_obj_id;
  }
}

void TransferService::HandleDataReceive(int client_fd,
                                        const ObjInfoHeader& header,
                                        bool is_respond_get_task) {
  LOG(INFO) << "Handling data receive for filename: " << header.dest_obj_id;
  if (header.obj_size <= 0) {
    LOG(ERROR) << "Invalid obj_size received: " << header.obj_size;
    if (is_respond_get_task) {
      ReportResult(header.task_id, false, "Invalid obj_size received");
    }
    SendErrorResponse(client_fd, header.task_id, header.dest_obj_id);
    return;
  }
  if (is_respond_get_task) {
    UpdateTaskMetrics(
        header.task_id, [obj_size = header.obj_size](TaskMetricContainer& ts) {
          if (auto* get_ts = dynamic_cast<GetTaskMetricContainer*>(&ts)) {
            get_ts->start_data_receiving_time = absl::Now();
            get_ts->data_size = obj_size;
          }
        });
  }
  std::string tmp_obj_id =
      std::string(header.dest_obj_id) + std::string(kTempFileSuffix);
  BufferObject buffer_obj(tmp_obj_id, header.obj_size,
                          /*overwrite=*/true);
  LOG(INFO) << "Successfully created buffer object";
  void* receiver_data_ptr = buffer_obj.get_data_ptr();

  if (!RecvAll(client_fd, receiver_data_ptr, header.obj_size).ok()) {
    LOG(ERROR) << "Failed to receive data for " << header.dest_obj_id;
    if (is_respond_get_task) {
      ReportResult(header.task_id, false, "Failed to receive data");
    }
    SendErrorResponse(client_fd, header.task_id, header.dest_obj_id);
    return;
  }

  // Close the buffer object to ensure data is flushed and the file descriptor
  // is released before renaming.
  buffer_obj.close();

  // Rename the temporary file to the final destination.
  if (rename(tmp_obj_id.c_str(), header.dest_obj_id) != 0) {
    PLOG(ERROR) << "Failed to rename temporary file " << tmp_obj_id << " to "
                << header.dest_obj_id;
    if (is_respond_get_task) {
      ReportResult(header.task_id, false, "Failed to rename temporary file");
    }
    SendErrorResponse(client_fd, header.task_id, header.dest_obj_id);
    return;
  }
  if (is_respond_get_task) {
    UpdateTaskMetrics(header.task_id, [](TaskMetricContainer& ts) {
      if (auto* get_ts = dynamic_cast<GetTaskMetricContainer*>(&ts)) {
        get_ts->data_received_time = absl::Now();
      }
    });
  }

  // Acknowledge the receipt of the data.
  ObjInfoHeader ack_header;
  ack_header.type = MessageType::kAck;
  ack_header.obj_size = 0;
  if (!SendAll(client_fd, &ack_header, kHeaderSize).ok()) {
    LOG(ERROR) << "Failed to send ACK for " << header.dest_obj_id;
    if (is_respond_get_task) {
      ReportResult(header.task_id, false, "Failed to send ACK");
    }
    return;
  }

  if (is_respond_get_task) {
    // This is the completing of a Get request. Update metric_container.
    UpdateTaskMetrics(header.task_id, [](TaskMetricContainer& ts) {
      ts.finish_time = absl::Now();
    });
    ReportResult(header.task_id, true, "Data received successfully!");
  }
}

void TransferService::ExecuteGetTask(GetTask* task) {
  auto& metric_container = task->GetMetricContainer();
  metric_container.start_execution_time = absl::Now();
  LOG(INFO) << "Executing GetTask for source address " << task->GetSourceAddr()
            << ", source obj id: " << task->GetSourceObjId();
  auto conn_opt = GetConnectionFromPool(task->GetSourceAddr());
  if (!conn_opt) {
    ReportResult(task->GetTaskId(), false, "Failed to get connection");
    return;
  }
  ScopedConnection conn = std::move(conn_opt.value());
  metric_container.connection_acquired_time = absl::Now();
  ObjInfoHeader header;
  header.type = MessageType::kGetObj;
  snprintf(header.task_id, sizeof(header.task_id), "%s",
           task->GetTaskId().c_str());
  snprintf(header.source_obj_id, sizeof(header.source_obj_id), "%s",
           task->GetSourceObjId().c_str());
  snprintf(header.dest_obj_id, sizeof(header.dest_obj_id), "%s",
           task->GetDestObjId().c_str());
  snprintf(header.source_address, sizeof(header.source_address), "%s",
           task->GetSourceAddr().c_str());
  snprintf(header.dest_address, sizeof(header.dest_address), "%s",
           task->GetDestAddr().c_str());
  header.obj_size = 0;  // Not used for request

  if (!SendAll(conn.fd(), &header, kHeaderSize).ok()) {
    LOG(ERROR) << "Failed to send GET_OBJ header";
    ReportResult(task->GetTaskId(), false, "Failed to send GET_OBJ header");
    return;
  }
  metric_container.header_sent_time = absl::Now();

  // Wait for the immediate response (ACK or ERROR)
  ObjInfoHeader resp_header;
  if (!RecvHeader(conn.fd(), resp_header).ok()) {
    ReportResult(task->GetTaskId(), false,
                 "Failed to receive response for GET request");
    return;
  }

  switch (resp_header.type) {
    case MessageType::kAck:
      LOG(INFO) << "Received ACK for GET request. Waiting for data transfer.";
      break;
    case MessageType::kError:
      ReportResult(task->GetTaskId(), false, "Received error message");
      break;
    case MessageType::kPutObj:
    case MessageType::kGetObj:
    case MessageType::kRespondToGetObj:
      ReportResult(task->GetTaskId(), false,
                   "Received unexpected response for GET request");
      break;
  }
}

void TransferService::ExecuteRespondToGetTask(RespondToGetTask* task) {
  auto& metric_container = task->GetMetricContainer();
  metric_container.start_execution_time = absl::Now();
  LOG(INFO) << "Executing RespondToGetTask for task_id=" << task->GetTaskId()
            << ", source_obj_id=" << task->GetSourceObjId()
            << ", dest_obj_id=" << task->GetDestObjId()
            << ", source_addr=" << task->GetSourceAddr()
            << ", dest_addr=" << task->GetDestAddr();

  auto conn_opt = GetConnectionFromPool(task->GetDestAddr());
  if (!conn_opt) {
    LOG(ERROR) << "Failed to get connection!";
    ReportResult(task->GetTaskId(), false, "Failed to get connection");
    return;
  }
  ScopedConnection conn = std::move(conn_opt.value());
  metric_container.connection_acquired_time = absl::Now();

  // Open file as buffer object
  BufferObject buffer_obj(task->GetSourceObjId());
  void* buffer_data_ptr = buffer_obj.get_data_ptr();
  size_t size = buffer_obj.get_capacity();

  if (buffer_data_ptr == nullptr) {
    LOG(ERROR) << "RespondToGetTask failed: Could not open buffer object for '"
               << task->GetSourceObjId() << "'";
    ReportResult(task->GetTaskId(), false, "Failed to create buffer object");
    return;
  }

  ObjInfoHeader header;

  std::memset(header.task_id, 0, sizeof(header.task_id));
  snprintf(header.task_id, sizeof(header.task_id), "%s",
           task->GetTaskId().c_str());
  header.task_id[sizeof(header.task_id) - 1] = '\0';

  std::memset(header.source_obj_id, 0, sizeof(header.source_obj_id));
  snprintf(header.source_obj_id, sizeof(header.source_obj_id), "%s",
           task->GetSourceObjId().c_str());

  std::memset(header.dest_obj_id, 0, sizeof(header.dest_obj_id));
  snprintf(header.dest_obj_id, sizeof(header.dest_obj_id), "%s",
           task->GetDestObjId().c_str());

  header.type = MessageType::kRespondToGetObj;
  header.obj_size = size;
  int sockfd = conn.fd();

  if (!SendAll(sockfd, &header, kHeaderSize).ok()) {
    LOG(ERROR) << "Failed to send kRespondToGetObj header";
    ReportResult(task->GetTaskId(), false, "Failed to send header");
    return;
  }
  metric_container.header_sent_time = absl::Now();

  if (!SendAll(sockfd, buffer_data_ptr, size).ok()) {
    LOG(ERROR) << "Failed to send buffer data";
    ReportResult(task->GetTaskId(), false, "Failed to send data");
    return;
  }
  metric_container.data_sent_time = absl::Now();

  ObjInfoHeader ack_header;
  if (!RecvHeader(sockfd, ack_header).ok()) {
    LOG(ERROR) << "Failed to receive ACK";
    ReportResult(task->GetTaskId(), false, "Failed to receive ACK");
    return;
  }

  if (ack_header.type != MessageType::kAck) {
    LOG(ERROR) << "Failed to receive ACK for RespondToGetObj";
    ReportResult(task->GetTaskId(), false, "Received unexpected ACK");
    return;
  }
  metric_container.finish_time = absl::Now();
  ReportResult(task->GetTaskId(), true,
               "RespondToGetTask completed successfully");
}

std::optional<ScopedConnection> TransferService::GetConnectionFromPool(
    const std::string& peer_addr) {
  auto pool = GetOrCreateConnectionPool(peer_addr);
  if (!pool) {
    LOG(ERROR) << "Failed to get or create connection pool for peer "
               << peer_addr;
    return std::nullopt;
  }
  auto conn_opt = pool->GetConnection();
  if (!conn_opt || !conn_opt->IsValid()) {
    LOG(ERROR) << "Failed to get connection from pool for peer '" << peer_addr
               << "'";
    return std::nullopt;
  }
  return conn_opt;
}

std::shared_ptr<ConnectionPool> TransferService::GetOrCreateConnectionPool(
    const std::string& peer_addr) {
  size_t colon_pos = peer_addr.find(':');
  if (colon_pos == std::string::npos) {
    LOG(ERROR) << "Invalid peer address format: " << peer_addr;
    return nullptr;
  }
  std::string peer_host = peer_addr.substr(0, colon_pos);
  int peer_port = std::stoi(peer_addr.substr(colon_pos + 1));
  if (peer_port <= 0 || peer_port > 65535) {
    LOG(ERROR) << "Invalid peer port: " << peer_port;
    return nullptr;
  }

  // Try read lock first
  {
    std::shared_lock read_lock(connection_pools_mutex_);
    auto it = connection_pools_.find(peer_addr);
    if (it != connection_pools_.end()) {
      return it->second;
    }
  }

  // Not found, need write lock
  std::unique_lock write_lock(connection_pools_mutex_);
  auto it = connection_pools_.find(peer_addr);
  if (it != connection_pools_.end()) {
    return it->second;  // Double check
  }

  // Pool doesn't exist, create it dynamically
  auto new_pool = std::make_shared<ConnectionPool>(peer_host, peer_port,
                                                   conn_pool_size_per_peer_);
  if (!new_pool->Initialize()) {
    LOG(ERROR) << "Failed to initialize new connection pool for target: "
               << peer_addr;
    return nullptr;
  }

  connection_pools_[peer_addr] = new_pool;
  return new_pool;
}

void TransferService::HandleGetObjRequest(int client_fd,
                                          const ObjInfoHeader& header) {
  LOG(INFO) << "Handling get object request for requested_obj_id: "
            << header.source_obj_id;

  if (!std::filesystem::exists(header.source_obj_id)) {
    PLOG(ERROR) << "Object not found: " << header.source_obj_id;
    SendErrorResponse(client_fd, header.task_id, header.source_obj_id);
    return;
  }

  // Send ACK to confirm request is accepted, before async data transfer
  ObjInfoHeader ack_header;
  ack_header.type = MessageType::kAck;
  snprintf(ack_header.task_id, sizeof(ack_header.task_id), "%s",
           header.task_id);
  if (!SendAll(client_fd, &ack_header, kHeaderSize).ok()) {
    LOG(ERROR) << "Failed to send ACK for GET request " << header.source_obj_id;
    return;  // Don't proceed if we can't even ACK
  }

  auto metric_container = std::make_shared<RespondToGetTaskMetricContainer>();
  metric_container->task_id = header.task_id;
  metric_container->task_type = TaskMetricContainer::TaskType::kRespondToGet;
  try {
    metric_container->data_size =
        std::filesystem::file_size(header.source_obj_id);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "Failed to get file size for metrics: " << e.what();
    metric_container->data_size = 0;
  }
  metric_container->submit_time = absl::Now();

  auto task = std::make_unique<RespondToGetTask>(
      header.task_id, header.source_obj_id, header.dest_obj_id,
      header.source_address, header.dest_address, metric_container);

  {
    std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
    // We don't have a promise for RespondToGetTask as it's triggered
    // remotely, but we want to track it.
    pending_tasks_[task->GetTaskId()] = {nullptr, metric_container};
  }

  task_queue_.enqueue(std::move(task));
}

void TransferService::UpdateTaskMetrics(
    const std::string& task_id,
    std::function<void(TaskMetricContainer&)> update_fn) {
  std::lock_guard<std::mutex> lock(pending_tasks_mutex_);
  auto it = pending_tasks_.find(task_id);
  if (it != pending_tasks_.end()) {
    update_fn(*(it->second.metric_container));
  }
}

}  // namespace ml_flashpoint::replication::transfer_service