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

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

TEST(TransferServiceTest, InitializeSucceedsWithFixedValidPorts) {
  constexpr int kNumThreads = 4;
  constexpr int kNumConnections = 10;

  TransferService service;

  // Use an ephemeral port (0) and let the OS assign an available port.
  int actual_port = service.Initialize(0, kNumThreads, kNumConnections);
  EXPECT_GT(actual_port, 0)
      << "Initialize failed to return a valid ephemeral port.";
}

TEST(TransferServiceTest, InitializeFailsWithInvalidPort) {
  TransferService service;
  EXPECT_DEATH(service.Initialize(-1, 4, 10),
               "listen_port must be non-negative");
}

TEST(TransferServiceTest, InitializeFailsWithInvalidThreads) {
  TransferService service;
  EXPECT_DEATH(service.Initialize(8080, 0, 10), "threads must be positive");
}

TEST(TransferServiceTest, InitializeFailsWithInvalidConnPool) {
  TransferService service;
  EXPECT_DEATH(service.Initialize(8080, 4, 0),
               "conn_pool_per_peer must be positive");
}

TEST(TransferServiceTest, InitializeSucceedsWithEphemeralPort) {
  TransferService service;
  EXPECT_GT(service.Initialize(0, 4, 10), 0);
}

TEST(TransferServiceTest, DoubleInitialize) {
  TransferService service;
  int port = service.Initialize();
  EXPECT_GT(port, 0);
  EXPECT_EQ(service.Initialize(), port);
}

TEST(TransferServiceTest, InitializeTwoServices) {
  TransferService service1;
  TransferService service2;
  int port1 = service1.Initialize();
  LOG(INFO) << "port1: " << port1;
  EXPECT_GT(port1, 0);
  int port2 = service2.Initialize();
  LOG(INFO) << "port2: " << port2;
  EXPECT_GT(port2, 0);
  EXPECT_NE(port1, port2);
}

TEST(TransferServiceTest, NormalShutdown) {
  TransferService service;
  service.Initialize();
  service.Shutdown();
  // Should not crash.
}

TEST(TransferServiceTest, ShutdownBeforeInitialize) {
  TransferService service;
  service.Shutdown();
  // Should not crash.
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
