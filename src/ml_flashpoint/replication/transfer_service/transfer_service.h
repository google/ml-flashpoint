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

#ifndef ML_FLASHPOINT_REPLICATION_TRANSFER_SERVICE_H_
#define ML_FLASHPOINT_REPLICATION_TRANSFER_SERVICE_H_

#include <future>
#include <optional>
#include <string>

// For socket programming
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#include "absl/status/statusor.h"
#include "connection_pool.h"
#include "mlf_log_sink.h"
#include "protocol.h"
#include "task.h"
#include "task_queue.h"
#include "thread_pool.h"

namespace ml_flashpoint::replication::transfer_service {

struct TransferResult {
  std::string task_id;
  bool success = false;
  std::string error_message;
};

class TransferService final {
 public:
  // Allow injecting a fake local_ip_address for unit testing. This must not be
  // exposed in public through bindings.cpp.
  explicit TransferService(
      const std::optional<std::string>& local_ip_address = std::nullopt);
  ~TransferService();

  // Initializes the transfer service.
  //
  // Args:
  //   listen_port: The port to listen on. If 0, an ephemeral port is chosen.
  //   threads: The number of worker threads in the thread pool.
  //   conn_pool_per_peer: The size of the connection pool for each peer.
  //   global_rank: The global rank of the process.
  //
  // Returns:
  //   The port the service is listening on, or -1 on failure.
  int Initialize(int listen_port = 0, int threads = 16,
                 int conn_pool_per_peer = 16, int global_rank = -1);

  // Shuts down the transfer service gracefully.
  void Shutdown();

  // Asynchronously puts data to a remote peer.
  //
  // Args:
  //   data_ptr: Pointer to the data in memory.
  //   data_size: Size of the data in bytes.
  //   dest_address: The destination service address, expected in "host:port"
  //                 format (e.g., "127.0.0.1:12345").
  //   dest_obj_id: A unique identifier for the object at the destination.
  //
  // Returns:
  //   A `std::future<TransferResult>` that will be set when the transfer
  //   completes.
  std::future<TransferResult> AsyncPut(void* data_ptr, size_t data_size,
                                       const std::string& dest_address,
                                       const std::string& dest_obj_id);

  // Asynchronously requests an object from the specified peer.
  //
  // Args:
  //   source_obj_id: The ID of the object to request from the remote peer.
  //   source_address: The source service address, expected in "host:port"
  //                   format (e.g., "127.0.0.1:12345").
  //   dest_obj_id: The ID to assign to the received object locally.
  //
  // Returns:
  //   A `std::future<TransferResult>` that will be set when the transfer
  //   completes.
  // Workflow of AsyncGet: AsyncGet[Service1] -> HandleGetObjRequest[Service2]
  // -> ExecuteRespondToGetTask[Service2] -> HandleDataReceive[Service1]
  std::future<TransferResult> AsyncGet(const std::string& source_obj_id,
                                       const std::string& source_address,
                                       const std::string& dest_obj_id);

 private:
  friend class PutTask;
  friend class GetTask;
  friend class RespondToGetTask;
  // Joins the given thread with a timeout.
  //
  // Args:
  //   t: The thread to join.
  //   timeout: The maximum duration to wait for the thread to join.
  //   thread_name: The name of the thread for logging purposes.
  void JoinThreadWithTimeout(std::thread& t, const absl::Duration timeout,
                             const std::string& thread_name);

  // Sets up the listening socket for the transfer service.
  // Returns the listener file descriptor, or an error status on failure.
  absl::StatusOr<int> SetupListeningSocket(int listen_port);

  void ProcessTaskQueueLoop();
  void ProcessEpollEventsLoop();
  void HandleNewConnection();
  void ProcessIncomingData(int client_fd);
  void RemoveClient(int client_fd);
  std::shared_ptr<ConnectionPool> GetOrCreateConnectionPool(
      const std::string& peer_addr);
  std::optional<ScopedConnection> GetConnectionFromPool(
      const std::string& peer_addr);
  void ReportResult(std::string task_id, bool success,
                    const std::string& message);

  void HandleDataReceive(int client_fd, const ObjInfoHeader& header,
                         bool is_respond_get_task);
  void HandleGetObjRequest(int client_fd, const ObjInfoHeader& header);

  void ExecutePutTask(PutTask* task);
  void ExecuteGetTask(GetTask* task);
  void ExecuteRespondToGetTask(RespondToGetTask* task);

  void SendErrorResponse(int client_fd, const char* task_id,
                         const char* dest_obj_id);

  int listen_port_ = 0;
  int threads_ = 0;
  int conn_pool_size_per_peer_ = 0;
  int listener_fd_ = -1;
  int epoll_fd_ = -1;
  std::optional<std::string>
      local_ip_address_;  // User input local ip address, only used for testing.
  std::string local_address_;  // Local address used in data transfer.
  int global_rank_ = -1;

  std::unique_ptr<ThreadPool> thread_pool_;
  std::thread epoll_thread_;
  std::thread task_queue_thread_;
  TaskQueue<TaskUniquePtr> task_queue_;

  std::atomic<bool> running_{false};
  std::mutex init_mutex_;  // Guard running_

  std::map<std::string, std::shared_ptr<ConnectionPool>> connection_pools_;
  mutable std::shared_mutex connection_pools_mutex_;  // Guard connection_pools_

  struct PendingTaskContext {
    std::shared_ptr<std::promise<TransferResult>> promise;
    std::shared_ptr<TaskMetricContainer> metric_container;
  };
  std::unordered_map<std::string, PendingTaskContext> pending_tasks_;
  std::mutex pending_tasks_mutex_;  // Guard pending_tasks_

  std::unique_ptr<MLFLogSink> mlf_log_sink_;

  void UpdateTaskMetrics(const std::string& task_id,
                         std::function<void(TaskMetricContainer&)> update_fn);
};

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_TRANSFER_SERVICE_H_
