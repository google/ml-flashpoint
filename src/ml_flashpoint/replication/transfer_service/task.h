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

#ifndef ML_FLASHPOINT_REPLICATION_TASK_H_
#define ML_FLASHPOINT_REPLICATION_TASK_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/time/time.h"

namespace ml_flashpoint::replication::transfer_service {

class TransferService;

class TaskMetricContainer {
 public:
  enum class TaskType { kUnknown, kPut, kGet, kRespondToGet };
  virtual ~TaskMetricContainer() = default;

  std::string task_id;
  TaskType task_type = TaskType::kUnknown;  // "Put" or "Get"
  size_t data_size = 0;                     // Size of data transferred
  absl::Time submit_time =
      absl::InfinitePast();  // When AsyncPut/Get was called
  absl::Time start_execution_time =
      absl::InfinitePast();  // When worker thread started executing
  absl::Time connection_acquired_time =
      absl::InfinitePast();  // When connection was acquired
  absl::Time header_sent_time =
      absl::InfinitePast();  // When header send completed
  absl::Time finish_time =
      absl::InfinitePast();  // When task execution finished

  virtual double WaitInQueueDurationMs() const;
  virtual double ConnectionAcquisitionDurationMs() const;
  virtual double HeaderSendingDurationMs() const;
  virtual double TotalDurationMs() const;

  virtual std::string ToString() const;

 protected:
  double SafeDiffMs(absl::Time start, absl::Time end,
                    const std::string& step_name) const;
};

class PutTaskMetricContainer : public TaskMetricContainer {
 public:
  absl::Time data_sent_time = absl::InfinitePast();

  double DataSendingDurationMs() const;
  std::string ToString() const override;
};

class GetTaskMetricContainer : public TaskMetricContainer {
 public:
  absl::Time start_data_receiving_time = absl::InfinitePast();
  absl::Time data_received_time = absl::InfinitePast();

  double DataReceivingDurationMs() const;
  std::string ToString() const override;
};

class RespondToGetTaskMetricContainer : public TaskMetricContainer {
 public:
  absl::Time data_sent_time = absl::InfinitePast();

  double DataSendingDurationMs() const;
  std::string ToString() const override;
};

// Returns the string representation of a TaskType.
std::string TaskTypeToString(TaskMetricContainer::TaskType type);

class Task {
 public:
  explicit Task(std::string task_id) : task_id_(std::move(task_id)) {}
  virtual ~Task() = default;

  virtual void Execute(TransferService* service) = 0;
  virtual TaskMetricContainer& GetMetricContainer() = 0;

  const std::string& GetTaskId() const { return task_id_; }

 private:
  const std::string task_id_;
};

using TaskUniquePtr = std::unique_ptr<Task>;

class PutTask : public Task {
 public:
  PutTask(std::string task_id, void* data_ptr, size_t data_size,
          std::string dest_obj_id, std::string dest_addr,
          std::shared_ptr<PutTaskMetricContainer> metric_container)
      : Task(std::move(task_id)),
        data_ptr_(data_ptr),
        data_size_(data_size),
        dest_obj_id_(std::move(dest_obj_id)),
        dest_addr_(std::move(dest_addr)),
        metric_container_(std::move(metric_container)) {}

  void Execute(TransferService* service) override;
  PutTaskMetricContainer& GetMetricContainer() override {
    return *metric_container_;
  }

  void* GetDataPtr() const { return data_ptr_; }
  size_t GetDataSize() const { return data_size_; }
  const std::string& GetDestObjId() const { return dest_obj_id_; }
  const std::string& GetDestAddr() const { return dest_addr_; }

 private:
  void* data_ptr_;
  size_t data_size_;
  std::string dest_obj_id_;
  std::string dest_addr_;
  std::shared_ptr<PutTaskMetricContainer> metric_container_;
};

class GetTask : public Task {
 public:
  GetTask(std::string task_id, std::string source_obj_id,
          std::string dest_obj_id, std::string source_addr,
          std::string dest_addr,
          std::shared_ptr<GetTaskMetricContainer> metric_container)
      : Task(std::move(task_id)),
        source_obj_id_(std::move(source_obj_id)),
        dest_obj_id_(std::move(dest_obj_id)),
        source_addr_(std::move(source_addr)),
        dest_addr_(std::move(dest_addr)),
        metric_container_(std::move(metric_container)) {}

  void Execute(TransferService* service) override;
  GetTaskMetricContainer& GetMetricContainer() override {
    return *metric_container_;
  }

  const std::string& GetSourceObjId() const { return source_obj_id_; }
  const std::string& GetDestObjId() const { return dest_obj_id_; }
  const std::string& GetSourceAddr() const { return source_addr_; }
  const std::string& GetDestAddr() const { return dest_addr_; }

 private:
  std::string source_obj_id_;
  std::string dest_obj_id_;
  std::string source_addr_;
  std::string dest_addr_;
  std::shared_ptr<GetTaskMetricContainer> metric_container_;
};

class RespondToGetTask : public Task {
 public:
  RespondToGetTask(
      std::string task_id, std::string source_obj_id, std::string dest_obj_id,
      std::string source_addr, std::string dest_addr,
      std::shared_ptr<RespondToGetTaskMetricContainer> metric_container)
      : Task(std::move(task_id)),
        source_obj_id_(std::move(source_obj_id)),
        dest_obj_id_(std::move(dest_obj_id)),
        source_addr_(std::move(source_addr)),
        dest_addr_(std::move(dest_addr)),
        metric_container_(std::move(metric_container)) {}

  void Execute(TransferService* service) override;
  RespondToGetTaskMetricContainer& GetMetricContainer() override {
    return *metric_container_;
  }

  const std::string& GetSourceObjId() const { return source_obj_id_; }
  const std::string& GetDestObjId() const { return dest_obj_id_; }
  const std::string& GetSourceAddr() const { return source_addr_; }
  const std::string& GetDestAddr() const { return dest_addr_; }

 private:
  std::string source_obj_id_;
  std::string dest_obj_id_;
  std::string source_addr_;
  std::string dest_addr_;
  std::shared_ptr<RespondToGetTaskMetricContainer> metric_container_;
};

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_TASK_H_
