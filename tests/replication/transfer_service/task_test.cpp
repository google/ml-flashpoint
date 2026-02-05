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

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

using ::testing::HasSubstr;

TEST(TaskMetricContainerTest, TaskTypeToString) {
  EXPECT_EQ(TaskTypeToString(TaskMetricContainer::TaskType::kPut), "Put");
  EXPECT_EQ(TaskTypeToString(TaskMetricContainer::TaskType::kGet), "Get");
  EXPECT_EQ(TaskTypeToString(TaskMetricContainer::TaskType::kRespondToGet),
            "RespondToGet");
  EXPECT_EQ(TaskTypeToString(TaskMetricContainer::TaskType::kUnknown),
            "Unknown");
}

TEST(TaskMetricContainerTest, SafeDiffMsHandlesInfinitePast) {
  TaskMetricContainer container;
  // start is InfinitePast
  EXPECT_EQ(container.WaitInQueueDurationMs(), 0.0);

  // end is InfinitePast
  container.submit_time = absl::Now();
  EXPECT_EQ(container.WaitInQueueDurationMs(), 0.0);
}

TEST(TaskMetricContainerTest, CalculationsAreCorrect) {
  TaskMetricContainer container;
  absl::Time start = absl::Now();
  container.submit_time = start;
  container.start_execution_time = start + absl::Milliseconds(10);
  container.connection_acquired_time = start + absl::Milliseconds(30);
  container.header_sent_time = start + absl::Milliseconds(60);
  container.finish_time = start + absl::Milliseconds(100);

  EXPECT_NEAR(container.WaitInQueueDurationMs(), 10.0, 0.001);
  EXPECT_NEAR(container.ConnectionAcquisitionDurationMs(), 20.0, 0.001);
  EXPECT_NEAR(container.HeaderSendingDurationMs(), 30.0, 0.001);
  EXPECT_NEAR(container.TotalDurationMs(), 100.0, 0.001);
}

TEST(PutTaskMetricContainerTest, DataSendingDuration) {
  PutTaskMetricContainer container;
  absl::Time start = absl::Now();
  container.header_sent_time = start;
  container.data_sent_time = start + absl::Milliseconds(50);

  EXPECT_NEAR(container.DataSendingDurationMs(), 50.0, 0.001);
}

TEST(GetTaskMetricContainerTest, DataReceivingDuration) {
  GetTaskMetricContainer container;
  absl::Time start = absl::Now();
  container.start_data_receiving_time = start;
  container.data_received_time = start + absl::Milliseconds(150);

  EXPECT_NEAR(container.DataReceivingDurationMs(), 150.0, 0.001);
}

TEST(RespondToGetTaskMetricContainerTest, DataSendingDuration) {
  RespondToGetTaskMetricContainer container;
  absl::Time start = absl::Now();
  container.header_sent_time = start;
  container.data_sent_time = start + absl::Milliseconds(75);

  EXPECT_NEAR(container.DataSendingDurationMs(), 75.0, 0.001);
}

TEST(RespondToGetTaskMetricContainerTest, ToStringContainsDurations) {
  RespondToGetTaskMetricContainer container;
  absl::Time start = absl::Now();
  container.submit_time = start;
  container.start_execution_time = start + absl::Milliseconds(10);
  container.connection_acquired_time = start + absl::Milliseconds(20);
  container.header_sent_time = start + absl::Milliseconds(30);
  container.data_sent_time = start + absl::Milliseconds(40);
  container.finish_time = start + absl::Milliseconds(50);

  std::string s = container.ToString();
  EXPECT_THAT(s, HasSubstr("wait_to_be_executed=10.000ms"));
  EXPECT_THAT(s, HasSubstr("connection_acquired=10.000ms"));
  EXPECT_THAT(s, HasSubstr("header_sent=10.000ms"));
  EXPECT_THAT(s, HasSubstr("data_sent=10.000ms"));
  EXPECT_THAT(s, HasSubstr("total=50.000ms"));
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
