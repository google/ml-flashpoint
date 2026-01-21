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

#include "task.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "transfer_service.h"

namespace ml_flashpoint::replication::transfer_service {

void PutTask::Execute(TransferService* service) {
  LOG(INFO) << "Executing PutTask for dest address " << dest_addr_;
  service->ExecutePutTask(this);
}

void GetTask::Execute(TransferService* service) {
  LOG(INFO) << "Executing GetTask for source address " << source_addr_;
  service->ExecuteGetTask(this);
}

void RespondToGetTask::Execute(TransferService* service) {
  LOG(INFO) << "Executing RespondToGetTask to " << dest_addr_;
  service->ExecuteRespondToGetTask(this);
}

double TaskMetricContainer::SafeDiffMs(absl::Time start, absl::Time end,
                                       const std::string& step_name) const {
  if (end == absl::InfinitePast() || start == absl::InfinitePast()) {
    if (finish_time != absl::InfinitePast() && !step_name.empty()) {
      LOG(WARNING) << "Task " << task_id << " finished but missing "
                   << step_name << " timestamp(s).";
    }
    return 0.0;
  }
  return absl::ToDoubleMilliseconds(end - start);
}

double TaskMetricContainer::WaitInQueueDurationMs() const {
  return SafeDiffMs(submit_time, start_execution_time, "wait");
}

double TaskMetricContainer::ConnectionAcquisitionDurationMs() const {
  return SafeDiffMs(start_execution_time, connection_acquired_time,
                    "connection");
}

double TaskMetricContainer::HeaderSendingDurationMs() const {
  return SafeDiffMs(connection_acquired_time, header_sent_time, "header_sent");
}

double TaskMetricContainer::TotalDurationMs() const {
  return SafeDiffMs(submit_time, finish_time, "total");
}

std::string TaskMetricContainer::ToString() const {
  return absl::StrFormat(
      "wait_to_be_executed=%.3fms, connection_acquired=%.3fms, "
      "header_sent=%.3fms, total=%.3fms",
      WaitInQueueDurationMs(), ConnectionAcquisitionDurationMs(),
      HeaderSendingDurationMs(), TotalDurationMs());
}

double PutTaskMetricContainer::DataSendingDurationMs() const {
  return SafeDiffMs(header_sent_time, data_sent_time, "data_sent");
}

std::string PutTaskMetricContainer::ToString() const {
  return absl::StrFormat(
      "wait_to_be_executed=%.3fms, connection_acquired=%.3fms, "
      "header_sent=%.3fms, data_sent=%.3fms, total=%.3fms",
      WaitInQueueDurationMs(), ConnectionAcquisitionDurationMs(),
      HeaderSendingDurationMs(), DataSendingDurationMs(), TotalDurationMs());
}

double GetTaskMetricContainer::DataReceivingDurationMs() const {
  return SafeDiffMs(start_data_receiving_time, data_received_time,
                    "data_received");
}

std::string GetTaskMetricContainer::ToString() const {
  return absl::StrFormat(
      "wait_to_be_executed=%.3fms, connection_acquired=%.3fms, "
      "header_sent=%.3fms, data_received=%.3fms, total=%.3fms",
      WaitInQueueDurationMs(), ConnectionAcquisitionDurationMs(),
      HeaderSendingDurationMs(), DataReceivingDurationMs(), TotalDurationMs());
}

double RespondToGetTaskMetricContainer::DataSendingDurationMs() const {
  return SafeDiffMs(header_sent_time, data_sent_time, "data_sent");
}

std::string RespondToGetTaskMetricContainer::ToString() const {
  return absl::StrFormat(
      "wait_to_be_executed=%.3fms, connection_acquired=%.3fms, "
      "header_sent=%.3fms, data_sent=%.3fms, total=%.3fms",
      WaitInQueueDurationMs(), ConnectionAcquisitionDurationMs(),
      HeaderSendingDurationMs(), DataSendingDurationMs(), TotalDurationMs());
}

std::string TaskTypeToString(TaskMetricContainer::TaskType type) {
  switch (type) {
    case TaskMetricContainer::TaskType::kPut:
      return "Put";
    case TaskMetricContainer::TaskType::kGet:
      return "Get";
    case TaskMetricContainer::TaskType::kRespondToGet:
      return "RespondToGet";
    default:
      return "Unknown";
  }
}

}  // namespace ml_flashpoint::replication::transfer_service
