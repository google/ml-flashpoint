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

#include "mlf_log_sink.h"

#include <filesystem>

#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace ml_flashpoint::replication::transfer_service {

void MLFLogSink::Send(const absl::LogEntry& entry) {
  // Format: [MLF YYYY-MM-DD HH:MM:SS,mss UTC LEVEL Rank=X filename:line]
  // message Example: [MLF 2025-12-10 20:15:15,951 UTC INFO Rank=0
  // transfer_service.cpp:134] TransferService::Shutdown

  std::string timestamp = absl::FormatTime(
      "%Y-%m-%d %H:%M:%S.%E3f %Z", entry.timestamp(), absl::UTCTimeZone());

  std::string severity_str = absl::LogSeverityName(entry.log_severity());

  std::string file_name =
      std::filesystem::path(entry.source_filename()).filename().string();

  absl::FPrintF(stderr, "[MLF %s %s Rank=%d %s:%d] %s\n", timestamp,
                severity_str, rank_, file_name, entry.source_line(),
                entry.text_message());
}

}  // namespace ml_flashpoint::replication::transfer_service
