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

#ifndef ML_FLASHPOINT_REPLICATION_TRANSFER_SERVICE_MLF_LOG_SINK_H_
#define ML_FLASHPOINT_REPLICATION_TRANSFER_SERVICE_MLF_LOG_SINK_H_

#include "absl/log/log_sink.h"

namespace ml_flashpoint::replication::transfer_service {

class MLFLogSink : public absl::LogSink {
 public:
  explicit MLFLogSink(int rank) : rank_(rank) {}
  void Send(const absl::LogEntry& entry) override;

 private:
  const int rank_;
};

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_TRANSFER_SERVICE_MLF_LOG_SINK_H_
