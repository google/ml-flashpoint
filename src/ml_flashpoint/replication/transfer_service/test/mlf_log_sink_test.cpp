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

#include <gtest/gtest.h>

#include <mutex>
#include <string>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "gmock/gmock.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

class MLFLogSinkTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    static std::once_flag flag;
    std::call_once(flag, []() { absl::InitializeLog(); });
  }

  void SetUp() override {
    original_threshold_ = absl::StderrThreshold();
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);
    testing::internal::CaptureStderr();
    captured_ = true;
  }

  void TearDown() override {
    absl::SetStderrThreshold(original_threshold_);
    if (captured_) {
      try {
        testing::internal::GetCapturedStderr();
      } catch (...) {
      }
    }
  }

  absl::LogSeverityAtLeast original_threshold_;
  bool captured_ = false;
};

TEST_F(MLFLogSinkTest, SendsInfoToStderr) {
  {
    MLFLogSink sink(10);
    absl::AddLogSink(&sink);
    LOG(INFO) << "Info message";
    absl::RemoveLogSink(&sink);
  }

  std::string output = testing::internal::GetCapturedStderr();
  captured_ = false;

  EXPECT_THAT(output, ::testing::StartsWith("[MLF "));
  EXPECT_THAT(output, ::testing::HasSubstr("Rank=10"));
  EXPECT_THAT(output, ::testing::HasSubstr("INFO"));
  EXPECT_THAT(output, ::testing::EndsWith("Info message\n"));
}

TEST_F(MLFLogSinkTest, SendsWarningToStderr) {
  {
    MLFLogSink sink(20);
    absl::AddLogSink(&sink);
    LOG(WARNING) << "Warning message";
    absl::RemoveLogSink(&sink);
  }

  std::string output = testing::internal::GetCapturedStderr();
  captured_ = false;

  EXPECT_THAT(output, ::testing::StartsWith("[MLF "));
  EXPECT_THAT(output, ::testing::HasSubstr("Rank=20"));
  EXPECT_THAT(output, ::testing::HasSubstr("WARNING"));
  EXPECT_THAT(output, ::testing::EndsWith("Warning message\n"));
}

TEST_F(MLFLogSinkTest, SendsErrorToStderr) {
  {
    MLFLogSink sink(30);
    absl::AddLogSink(&sink);
    LOG(ERROR) << "Error message";
    absl::RemoveLogSink(&sink);
  }

  std::string output = testing::internal::GetCapturedStderr();
  captured_ = false;

  EXPECT_THAT(output, ::testing::StartsWith("[MLF "));
  EXPECT_THAT(output, ::testing::HasSubstr("Rank=30"));
  EXPECT_THAT(output, ::testing::HasSubstr("ERROR"));
  EXPECT_THAT(output, ::testing::EndsWith("Error message\n"));
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service