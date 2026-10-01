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

#include <arpa/inet.h>
#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "gtest/gtest.h"
#include "net_util.h"
#include "protocol.h"
#include "transfer_service.h"

namespace ml_flashpoint::replication::transfer_service {

namespace {

class TestLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    std::lock_guard<std::mutex> lock(mutex_);
    messages.push_back(std::string(entry.text_message()));
  }
  std::vector<std::string> messages;
  std::mutex mutex_;
};

// Helper function to verify monotonic timestamps.
void ValidateTaskTimestamps(const TaskMetricContainer& ts) {
  EXPECT_NE(ts.submit_time, absl::InfinitePast());
  EXPECT_NE(ts.start_execution_time, absl::InfinitePast());
  EXPECT_NE(ts.connection_acquired_time, absl::InfinitePast());
  EXPECT_NE(ts.header_sent_time, absl::InfinitePast());
  EXPECT_NE(ts.finish_time, absl::InfinitePast());

  EXPECT_LE(ts.submit_time, ts.start_execution_time);
  EXPECT_LE(ts.start_execution_time, ts.connection_acquired_time);
  EXPECT_LE(ts.connection_acquired_time, ts.header_sent_time);

  if (ts.task_type == TaskMetricContainer::TaskType::kPut) {
    const auto* put_ts = dynamic_cast<const PutTaskMetricContainer*>(&ts);
    ASSERT_NE(put_ts, nullptr);
    EXPECT_NE(put_ts->data_sent_time, absl::InfinitePast());
    EXPECT_NE(ts.finish_time, absl::InfinitePast());

    EXPECT_LE(ts.header_sent_time, put_ts->data_sent_time);
    EXPECT_LE(put_ts->data_sent_time, ts.finish_time);
  } else if (ts.task_type == TaskMetricContainer::TaskType::kGet) {
    const auto* get_ts = dynamic_cast<const GetTaskMetricContainer*>(&ts);
    ASSERT_NE(get_ts, nullptr);
    EXPECT_NE(get_ts->start_data_receiving_time, absl::InfinitePast());
    EXPECT_NE(get_ts->data_received_time, absl::InfinitePast());
    EXPECT_LE(ts.header_sent_time, get_ts->start_data_receiving_time);
    EXPECT_LE(get_ts->start_data_receiving_time, get_ts->data_received_time);

    EXPECT_LE(get_ts->data_received_time, ts.finish_time);
  } else if (ts.task_type == TaskMetricContainer::TaskType::kRespondToGet) {
    const auto* respond_ts =
        dynamic_cast<const RespondToGetTaskMetricContainer*>(&ts);
    ASSERT_NE(respond_ts, nullptr);
    EXPECT_NE(respond_ts->data_sent_time, absl::InfinitePast());
    EXPECT_LE(ts.header_sent_time, respond_ts->data_sent_time);
    EXPECT_LE(respond_ts->data_sent_time, ts.finish_time);
  }
}

// Helper function to verify file content and then remove the file.
void VerifyFileContentAndRemove(const std::string& file_path,
                                const std::string& expected_content) {
  std::ifstream input_file(file_path);
  ASSERT_TRUE(input_file.is_open())
      << "Failed to open test file: " << file_path;
  std::stringstream buffer;
  buffer << input_file.rdbuf();
  const std::string received_data = buffer.str();
  EXPECT_EQ(expected_content, received_data);
  input_file.close();
  std::remove(file_path.c_str());
}

TEST(TransferServiceP2PTest, SimplePut) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "Hello, world!";
  std::string obj_id = "my_object_simple";

  auto put_future =
      service1.AsyncPut((void*)data.c_str(), data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);

  VerifyFileContentAndRemove(obj_id, data);

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutLargeObject) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  // Create a large string (5MB)
  const size_t large_size = 5 * 1024 * 1024;
  std::string large_data(large_size, 'A');
  for (size_t i = 0; i < large_size; ++i) {
    large_data[i] = 'A' + (i % 26);
  }

  std::string obj_id = "my_large_object";

  auto put_future =
      service1.AsyncPut((void*)large_data.c_str(), large_data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);

  VerifyFileContentAndRemove(obj_id, large_data);

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, ShutdownInterruptsTransfer) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  // Create a large string (100MB) to make sure it takes time to send
  const size_t large_size = 100 * 1024 * 1024;
  std::string large_data(large_size, 'A');

  std::string obj_id = "my_interrupt_object";

  auto put_future =
      service1.AsyncPut((void*)large_data.c_str(), large_data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);

  // Wait a small amount of time to let the transfer start
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Trigger shutdown!
  service1.Shutdown();

  try {
    auto put_result = put_future.get();
    EXPECT_FALSE(put_result.success);
    LOG(INFO) << "Transfer failed as expected after shutdown.";
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(e.what(), testing::HasSubstr("Service is shutting down"));
    LOG(INFO) << "Transfer threw exception as expected after shutdown: "
              << e.what();
  }

  // Cleanup file if it was partially created
  std::remove(obj_id.c_str());
  std::remove((obj_id + ".tmp").c_str());

  service2.Shutdown();
}

TEST(TransferServiceP2PTest, AsyncPutLargeMmapData) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  const size_t large_size = 1UL * 1024 * 1024 * 1024;  // 1 GB
  const std::string obj_id = "my_large_mmap_object";
  const std::string temp_file_path = "large_mmap_file.tmp";

  // Create and setup a large file.
  int fd = open(temp_file_path.c_str(), O_RDWR | O_CREAT, 0666);
  ASSERT_NE(fd, -1);
  ASSERT_NE(ftruncate(fd, large_size), -1);

  // mmap the file.
  void* mapped_data =
      mmap(NULL, large_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  ASSERT_NE(mapped_data, MAP_FAILED);

  // Fill the mmaped region with some data.
  char* data_ptr = static_cast<char*>(mapped_data);
  for (size_t i = 0; i < large_size; ++i) {
    data_ptr[i] = 'A' + (i % 26);
  }

  auto put_future = service1.AsyncPut(
      mapped_data, large_size, "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);

  // Verify the received file.
  std::ifstream received_file(obj_id, std::ios::binary);
  ASSERT_TRUE(received_file.is_open());
  std::vector<char> received_buffer(1024 * 1024);
  size_t total_read = 0;
  while (received_file) {
    received_file.read(received_buffer.data(), received_buffer.size());
    size_t read_count = received_file.gcount();
    for (size_t i = 0; i < read_count; ++i) {
      ASSERT_EQ(received_buffer[i], 'A' + ((total_read + i) % 26));
    }
    total_read += read_count;
  }
  ASSERT_EQ(total_read, large_size);

  // Cleanup.
  munmap(mapped_data, large_size);
  close(fd);
  std::remove(temp_file_path.c_str());
  std::remove(obj_id.c_str());
  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutEmptyObjectShouldFail) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "";
  std::string obj_id = "my_empty_object";

  auto put_future =
      service1.AsyncPut((void*)data.c_str(), data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  EXPECT_THROW(
      {
        try {
          put_future.get();
        } catch (const std::runtime_error& e) {
          LOG(INFO) << "Caught expected exception for empty object put: "
                    << e.what();
          throw;  // Re-throw to satisfy EXPECT_THROW
        }
      },
      std::runtime_error);

  // Ensure the file was not created or is cleaned up.
  std::remove(obj_id.c_str());
  std::ifstream input_file(obj_id);
  EXPECT_FALSE(input_file.is_open());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutZeroSizeObjectShouldFail) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "some data";
  std::string obj_id = "my_negative_object";

  // AsyncPut takes size_t, so we can't easily pass negative here from C++.
  // But we want to test HandleDataReceive's behavior if it receives negative
  // size in header. We can simulate this by manually sending a malformed header
  // if we had lower level access, but here we can at least test that if we pass
  // 0 it fails.

  auto put_future = service1.AsyncPut(
      (void*)data.c_str(), 0, "127.0.0.1:" + std::to_string(port2), obj_id);
  EXPECT_THROW(put_future.get(), std::runtime_error);

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, ConcurrentPut) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  const int num_threads = 8;
  std::vector<std::thread> threads;
  std::vector<std::string> object_ids(num_threads);
  std::vector<std::string> data_payloads(num_threads);
  std::vector<std::future<TransferResult>> futures(num_threads);

  auto put_task = [&](int i) {
    data_payloads[i] = "Concurrent data " + std::to_string(i);
    object_ids[i] = "concurrent_object_" + std::to_string(i);
    futures[i] = service1.AsyncPut(
        (void*)data_payloads[i].c_str(), data_payloads[i].size(),
        "127.0.0.1:" + std::to_string(port2), object_ids[i]);
  };

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(put_task, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < num_threads; ++i) {
    auto result = futures[i].get();
    EXPECT_TRUE(result.success);
    VerifyFileContentAndRemove(object_ids[i], data_payloads[i]);
  }

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, SimpleGet) {
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2(std::optional<std::string>("127.0.0.1"));
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "Hello, world!";
  std::string obj_id = "my_object_simple_get";
  std::string dest_obj_id = "my_object_simple_get_local";

  // service1 "owns" the object, create a file for it.
  std::ofstream out_file(obj_id);
  out_file << data;
  out_file.close();

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);

  // std::this_thread::sleep_for(std::chrono::seconds(15));
  auto get_result = get_future.get();
  EXPECT_TRUE(get_result.success);

  VerifyFileContentAndRemove(dest_obj_id, data);
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetLargeObject) {
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2(std::optional<std::string>("127.0.0.1"));
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  const size_t large_size = 5 * 1024 * 1024;
  std::string large_data(large_size, 'A');
  for (size_t i = 0; i < large_size; ++i) {
    large_data[i] = 'A' + (i % 26);
  }

  std::string obj_id = "my_large_object_get";
  std::string dest_obj_id = "my_large_object_get_local";

  std::ofstream out_file(obj_id, std::ios::binary);
  out_file.write(large_data.c_str(), large_data.size());
  out_file.close();

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  try {
    auto get_result = get_future.get();
    EXPECT_TRUE(get_result.success);
  } catch (const std::future_error& e) {
    FAIL() << "Test failed with std::future_error: " << e.what()
           << ". This likely means the GetTask was destroyed prematurely, "
              "breaking the promise.";
  }

  VerifyFileContentAndRemove(dest_obj_id, large_data);
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, ConcurrentGet) {
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2(std::optional<std::string>("127.0.0.1"));
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  const int num_threads = 8;
  std::vector<std::thread> threads;
  std::vector<std::string> object_ids(num_threads);
  std::vector<std::string> local_object_ids(num_threads);
  std::vector<std::string> data_payloads(num_threads);
  std::vector<std::future<TransferResult>> futures(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    data_payloads[i] = "Concurrent data " + std::to_string(i);
    object_ids[i] = "concurrent_object_get_" + std::to_string(i);
    local_object_ids[i] = "concurrent_object_get_local_" + std::to_string(i);
    std::ofstream out_file(object_ids[i]);
    out_file << data_payloads[i];
    out_file.close();
  }

  auto get_task = [&](int i) {
    futures[i] =
        service2.AsyncGet(object_ids[i], "127.0.0.1:" + std::to_string(port1),
                          local_object_ids[i]);
  };

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(get_task, i);
  }

  for (auto& t : threads) {
    t.join();
  }

  for (int i = 0; i < num_threads; ++i) {
    auto result = futures[i].get();
    EXPECT_TRUE(result.success);
    VerifyFileContentAndRemove(local_object_ids[i], data_payloads[i]);
    std::remove(object_ids[i].c_str());
  }

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetNonExistentObjectShouldFail) {
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2(std::optional<std::string>("127.0.0.1"));
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string obj_id = "non_existent_object";
  std::string dest_obj_id = "non_existent_object_local";

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);

  EXPECT_THROW(
      {
        try {
          get_future.get();
        } catch (const std::runtime_error& e) {
          LOG(INFO) << "Caught expected exception for non-existent object get: "
                    << e.what();
          EXPECT_STREQ(e.what(), "Received error message");
          throw;
        }
      },
      std::runtime_error);

  std::ifstream input_file(dest_obj_id);
  EXPECT_FALSE(input_file.is_open());
  std::remove(dest_obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

// ---------------------------------------------------------------------------
// Helpers for driving a TransferService with a raw client socket, used by the
// dest_address (SSRF) tests below.
// ---------------------------------------------------------------------------

// Connects a blocking TCP socket to 127.0.0.1:port. Returns -1 on failure.
int ConnectToLocalPort(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

// Creates a blocking listener on 127.0.0.1 with an ephemeral port. Returns the
// listener fd and sets `port` to the bound port, or returns -1 on failure.
int ListenOnLocalEphemeralPort(int* port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = 0;
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  socklen_t addr_len = sizeof(addr);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), addr_len) != 0 ||
      getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) != 0 ||
      listen(fd, 16) != 0) {
    close(fd);
    return -1;
  }
  *port = ntohs(addr.sin_port);
  return fd;
}

void WriteTestFile(const std::string& path, const std::string& data) {
  std::ofstream out_file(path);
  out_file << data;
  out_file.close();
}

ObjInfoHeader BuildGetRequest(const std::string& task_id,
                              const std::string& source_obj_id,
                              const std::string& dest_obj_id,
                              const std::string& dest_address) {
  ObjInfoHeader request;
  request.type = MessageType::kGetObj;
  snprintf(request.task_id, sizeof(request.task_id), "%s", task_id.c_str());
  snprintf(request.source_obj_id, sizeof(request.source_obj_id), "%s",
           source_obj_id.c_str());
  snprintf(request.dest_obj_id, sizeof(request.dest_obj_id), "%s",
           dest_obj_id.c_str());
  snprintf(request.dest_address, sizeof(request.dest_address), "%s",
           dest_address.c_str());
  return request;
}

// Sends `request` on `client_fd` and returns the immediate response header.
ObjInfoHeader SendRequestAndRecvResponse(int client_fd,
                                         const ObjInfoHeader& request) {
  ObjInfoHeader response;
  EXPECT_TRUE(SendAll(client_fd, &request, kHeaderSize).ok());
  EXPECT_TRUE(RecvHeader(client_fd, response).ok());
  return response;
}

// Waits for the service to connect to `listener`, then consumes exactly one
// kRespondToGetObj exchange on the accepted connection: validates the header
// and payload, and sends back a kAck so the responder completes cleanly.
void ExpectGetResponseOnListener(int listener,
                                 const std::string& expected_task_id,
                                 const std::string& expected_data) {
  pollfd pfd{listener, POLLIN, 0};
  ASSERT_GT(poll(&pfd, 1, /*timeout_ms=*/5000), 0)
      << "Service did not connect back to the requester's address";
  int response_fd = accept(listener, nullptr, nullptr);
  ASSERT_GE(response_fd, 0);

  ObjInfoHeader response;
  ASSERT_TRUE(RecvHeader(response_fd, response).ok());
  EXPECT_EQ(response.type, MessageType::kRespondToGetObj);
  EXPECT_EQ(response.obj_size, static_cast<ssize_t>(expected_data.size()));
  EXPECT_STREQ(response.task_id, expected_task_id.c_str());

  std::string actual_data(expected_data.size(), '\0');
  ASSERT_TRUE(
      RecvAll(response_fd, actual_data.data(), expected_data.size()).ok());
  EXPECT_EQ(actual_data, expected_data);

  ObjInfoHeader data_ack;
  data_ack.type = MessageType::kAck;
  EXPECT_TRUE(SendAll(response_fd, &data_ack, kHeaderSize).ok());
  close(response_fd);
}

// Serves kRespondToGetObj exchanges arriving on `listener` (over any number
// of connections) until `expected_count` have been handled or the deadline
// passes. Returns the task_ids that were served.
std::vector<std::string> ServeGetResponses(int listener, size_t expected_count,
                                           const std::string& expected_data) {
  std::vector<std::string> served_task_ids;
  std::vector<pollfd> fds = {{listener, POLLIN, 0}};
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);

  while (served_task_ids.size() < expected_count &&
         std::chrono::steady_clock::now() < deadline) {
    int ret = poll(fds.data(), fds.size(), /*timeout_ms=*/200);
    if (ret <= 0) continue;

    for (size_t i = 0; i < fds.size(); ++i) {
      if (!(fds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      if (fds[i].fd == listener) {
        int accepted = accept(listener, nullptr, nullptr);
        if (accepted >= 0) fds.push_back({accepted, POLLIN, 0});
        continue;
      }
      ObjInfoHeader response;
      if (!RecvHeader(fds[i].fd, response).ok()) {
        close(fds[i].fd);
        fds.erase(fds.begin() + i);
        --i;
        continue;
      }
      EXPECT_EQ(response.type, MessageType::kRespondToGetObj);
      std::string actual_data(response.obj_size, '\0');
      EXPECT_TRUE(
          RecvAll(fds[i].fd, actual_data.data(), response.obj_size).ok());
      EXPECT_EQ(actual_data, expected_data);
      ObjInfoHeader data_ack;
      data_ack.type = MessageType::kAck;
      EXPECT_TRUE(SendAll(fds[i].fd, &data_ack, kHeaderSize).ok());
      served_task_ids.push_back(response.task_id);
    }
  }
  for (size_t i = 1; i < fds.size(); ++i) close(fds[i].fd);
  return served_task_ids;
}

// ---------------------------------------------------------------------------
// dest_address (SSRF) tests. A kGetObj request must never be able to direct
// the responding service to connect to an arbitrary host; the response must
// always go back to the address the request actually came from.
// ---------------------------------------------------------------------------

class SpoofedDestHostTest : public ::testing::TestWithParam<std::string> {};

// The host portion of dest_address is ignored regardless of what it contains;
// the response is delivered to the requester's real address.
TEST_P(SpoofedDestHostTest, GetResponseGoesBackToRequester) {
  // Given: a service that owns an object, and a "requester" listener that
  // stands in for the requesting service's listening port.
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string data = "Hello, world!";
  const std::string obj_id = "my_object_spoofed_dest";
  WriteTestFile(obj_id, data);
  int requester_port = 0;
  int requester_listener = ListenOnLocalEphemeralPort(&requester_port);
  ASSERT_GE(requester_listener, 0);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When: a kGetObj names the spoofed host but the requester's real port.
  const std::string spoofed_dest_address =
      GetParam() + ":" + std::to_string(requester_port);
  ObjInfoHeader ack = SendRequestAndRecvResponse(
      client_fd, BuildGetRequest("spoofed_task", obj_id, obj_id + "_local",
                                 spoofed_dest_address));

  // Then: the request is accepted and the response arrives at 127.0.0.1.
  EXPECT_EQ(ack.type, MessageType::kAck);
  ASSERT_NO_FATAL_FAILURE(
      ExpectGetResponseOnListener(requester_listener, "spoofed_task", data));

  close(requester_listener);
  close(client_fd);
  std::remove(obj_id.c_str());
  service1.Shutdown();
}

INSTANTIATE_TEST_SUITE_P(TransferServiceP2PTest, SpoofedDestHostTest,
                         ::testing::Values("203.0.113.1",      // TEST-NET-3
                                           "10.0.0.1",         // private
                                           "169.254.169.254",  // metadata
                                           "0.0.0.0", "255.255.255.255",
                                           "not-a-host", "127.0.0.1",
                                           /*empty host=*/""));

class InvalidDestAddressTest : public ::testing::TestWithParam<std::string> {};

// A dest_address without a usable port is rejected with kError before the
// request is acknowledged, so no connection attempt is ever made.
TEST_P(InvalidDestAddressTest, GetRequestIsRejected) {
  // Given: a service that owns an object.
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);
  const std::string obj_id = "my_object_invalid_dest";
  WriteTestFile(obj_id, "data");
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When: a kGetObj is sent with the invalid dest_address.
  ObjInfoHeader response = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("invalid_task", obj_id, obj_id + "_local", GetParam()));

  // Then: the service replies with an error instead of an ACK.
  EXPECT_EQ(response.type, MessageType::kError);
  EXPECT_STREQ(response.task_id, "invalid_task");

  close(client_fd);
  std::remove(obj_id.c_str());
  service1.Shutdown();
}

INSTANTIATE_TEST_SUITE_P(TransferServiceP2PTest, InvalidDestAddressTest,
                         ::testing::Values("", "not-an-address", "127.0.0.1",
                                           "127.0.0.1:", ":", "127.0.0.1:0",
                                           "127.0.0.1:65536", "127.0.0.1:-1",
                                           "127.0.0.1:abc", "127.0.0.1:12ab"));

// A rejected request must not poison the connection: the same client socket
// can still issue a valid request afterwards.
TEST(TransferServiceP2PTest, GetRequestRejectionKeepsConnectionUsable) {
  // Given: a service with an object, a requester listener and a client.
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string data = "still alive";
  const std::string obj_id = "my_object_after_rejection";
  WriteTestFile(obj_id, data);
  int requester_port = 0;
  int requester_listener = ListenOnLocalEphemeralPort(&requester_port);
  ASSERT_GE(requester_listener, 0);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When: an invalid request is followed by a valid one on the same socket.
  ObjInfoHeader first = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("bad_task", obj_id, obj_id + "_local", "127.0.0.1:0"));
  ObjInfoHeader second = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("good_task", obj_id, obj_id + "_local",
                      "127.0.0.1:" + std::to_string(requester_port)));

  // Then: the first is rejected, the second is served normally.
  EXPECT_EQ(first.type, MessageType::kError);
  EXPECT_STREQ(first.task_id, "bad_task");
  EXPECT_EQ(second.type, MessageType::kAck);
  ASSERT_NO_FATAL_FAILURE(
      ExpectGetResponseOnListener(requester_listener, "good_task", data));

  close(requester_listener);
  close(client_fd);
  std::remove(obj_id.c_str());
  service1.Shutdown();
}

// A dest_address that fills the whole buffer without a NUL terminator is
// handled safely (no over-read, no crash) and rejected.
TEST(TransferServiceP2PTest, GetRequestWithUnterminatedDestAddressIsRejected) {
  // Given: a service with an object and a connected client.
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);
  const std::string obj_id = "my_object_unterminated_dest";
  WriteTestFile(obj_id, "data");
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When: dest_address is 64 non-NUL bytes.
  ObjInfoHeader request =
      BuildGetRequest("unterminated_task", obj_id, obj_id + "_local", "");
  std::memset(request.dest_address, 'A', sizeof(request.dest_address));
  ObjInfoHeader response = SendRequestAndRecvResponse(client_fd, request);

  // Then: the request is rejected and the service is still responsive.
  EXPECT_EQ(response.type, MessageType::kError);
  EXPECT_STREQ(response.task_id, "unterminated_task");
  ObjInfoHeader probe = SendRequestAndRecvResponse(
      client_fd, BuildGetRequest("probe_task", obj_id, obj_id + "_local",
                                 "not-an-address"));
  EXPECT_EQ(probe.type, MessageType::kError);
  EXPECT_STREQ(probe.task_id, "probe_task");

  close(client_fd);
  std::remove(obj_id.c_str());
  service1.Shutdown();
}

// Multiple concurrent clients all spoofing different hosts each get their
// response delivered to the real requester address.
TEST(TransferServiceP2PTest, ConcurrentSpoofedGetRequestsAllReturnToRequester) {
  // Given: a service with an object, a shared requester listener, and
  // several connected clients.
  constexpr int kNumClients = 4;
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/8,
                                  /*conn_pool_per_peer=*/kNumClients);
  ASSERT_GT(port1, 0);
  const std::string data = "concurrent payload";
  const std::string obj_id = "my_object_concurrent_spoof";
  WriteTestFile(obj_id, data);
  int requester_port = 0;
  int requester_listener = ListenOnLocalEphemeralPort(&requester_port);
  ASSERT_GE(requester_listener, 0);
  std::vector<int> client_fds;
  for (int i = 0; i < kNumClients; ++i) {
    client_fds.push_back(ConnectToLocalPort(port1));
    ASSERT_GE(client_fds.back(), 0);
  }

  // When: every client sends a spoofed kGetObj at the same time.
  std::vector<std::string> expected_task_ids;
  for (int i = 0; i < kNumClients; ++i) {
    expected_task_ids.push_back("concurrent_task_" + std::to_string(i));
  }
  std::vector<std::thread> senders;
  std::vector<ObjInfoHeader> acks(kNumClients);
  for (int i = 0; i < kNumClients; ++i) {
    senders.emplace_back([&, i] {
      acks[i] = SendRequestAndRecvResponse(
          client_fds[i],
          BuildGetRequest(expected_task_ids[i], obj_id, obj_id + "_local",
                          "10.0.0." + std::to_string(i + 1) + ":" +
                              std::to_string(requester_port)));
    });
  }
  std::vector<std::string> actual_task_ids =
      ServeGetResponses(requester_listener, kNumClients, data);
  for (auto& t : senders) t.join();

  // Then: every request was ACKed and every response reached the requester.
  for (const auto& ack : acks) EXPECT_EQ(ack.type, MessageType::kAck);
  std::sort(actual_task_ids.begin(), actual_task_ids.end());
  EXPECT_EQ(actual_task_ids, expected_task_ids);

  close(requester_listener);
  for (int fd : client_fds) close(fd);
  std::remove(obj_id.c_str());
  service1.Shutdown();
}

// Happy path through the public API: because the responder connects back to
// the address the request came from, a Get succeeds even when the requesting
// service advertises a local address that is not reachable.
TEST(TransferServiceP2PTest, GetSucceedsWhenRequesterAdvertisesUnreachableIp) {
  // Given: service1 owns an object; service2 advertises an unroutable IP.
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);
  TransferService service2(std::optional<std::string>("203.0.113.5"));
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);
  const std::string data = "reachable after all";
  const std::string obj_id = "my_object_unreachable_requester";
  const std::string dest_obj_id = obj_id + "_local";
  WriteTestFile(obj_id, data);

  // When: service2 requests the object from service1 over loopback.
  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  TransferResult actual_result = get_future.get();

  // Then: the transfer completes and the data is correct.
  EXPECT_TRUE(actual_result.success);
  VerifyFileContentAndRemove(dest_obj_id, data);

  std::remove(obj_id.c_str());
  service1.Shutdown();
  service2.Shutdown();
}

// Shutting down while a remotely triggered RespondToGetTask is still trying
// to connect back must not crash (such tasks have no promise to fail).
TEST(TransferServiceP2PTest, ShutdownWithInFlightRespondToGetTaskIsSafe) {
  // Given: a service with an object, and a requester port nobody listens on.
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string obj_id = "my_object_shutdown_in_flight";
  WriteTestFile(obj_id, "data");
  int closed_port = 0;
  int tmp_listener = ListenOnLocalEphemeralPort(&closed_port);
  ASSERT_GE(tmp_listener, 0);
  close(tmp_listener);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When: a GET is accepted (so a RespondToGetTask is queued and starts
  // retrying the connection) and the service is shut down immediately.
  ObjInfoHeader ack = SendRequestAndRecvResponse(
      client_fd, BuildGetRequest("in_flight_task", obj_id, obj_id + "_local",
                                 "127.0.0.1:" + std::to_string(closed_port)));
  EXPECT_EQ(ack.type, MessageType::kAck);
  service1.Shutdown();

  // Then: the process is still alive and a new service can be started.
  TransferService service2(std::optional<std::string>("127.0.0.1"));
  EXPECT_GT(service2.Initialize(), 0);
  service2.Shutdown();

  close(client_fd);
  std::remove(obj_id.c_str());
}

TEST(TransferServiceP2PTest, PutCreatesTemporaryFileAndRenames) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "Temporary file test data.";
  std::string obj_id = "my_object_temp_test";
  std::string tmp_obj_id = obj_id + ".tmp";

  // Ensure files don't exist before the test.
  std::remove(obj_id.c_str());
  std::remove(tmp_obj_id.c_str());

  auto put_future =
      service1.AsyncPut((void*)data.c_str(), data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);

  // After completion, the temporary file should not exist.
  std::ifstream temp_file(tmp_obj_id);
  EXPECT_FALSE(temp_file.is_open());

  // The final file should exist and have the correct content.
  VerifyFileContentAndRemove(obj_id, data);

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetCreatesTemporaryFileAndRenames) {
  TransferService service1(std::optional<std::string>("127.0.0.1"));
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2(std::optional<std::string>("127.0.0.1"));
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "Temporary file test data for GET.";
  std::string obj_id = "my_object_temp_test_get_source";
  std::string dest_obj_id = "my_object_temp_test_get_dest";
  std::string tmp_dest_obj_id = dest_obj_id + ".tmp";

  // Create the source file on service1's side.
  std::ofstream out_file(obj_id);
  ASSERT_TRUE(out_file.is_open());
  out_file << data;
  out_file.close();

  // Ensure destination files don't exist before the test.
  std::remove(dest_obj_id.c_str());
  std::remove(tmp_dest_obj_id.c_str());

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  auto get_result = get_future.get();
  EXPECT_TRUE(get_result.success);

  // After completion, the temporary file should not exist on the destination.
  std::ifstream temp_file(tmp_dest_obj_id);
  EXPECT_FALSE(temp_file.is_open());

  // The final file should exist on the destination and have the correct
  // content.
  VerifyFileContentAndRemove(dest_obj_id, data);
  // Clean up the source file.
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutOverwritesEmptyTempFile) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "some data here";
  std::string obj_id = "some_obj";
  std::string tmp_obj_id = obj_id + ".tmp";
  std::ofstream(tmp_obj_id).close();  // Create empty temp file

  auto put_future =
      service1.AsyncPut((void*)data.c_str(), data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);
  VerifyFileContentAndRemove(obj_id, data);
  EXPECT_FALSE(std::ifstream(tmp_obj_id).good());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutOverwritesNonEmptyTempFile) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string old_data = "old data";
  std::string new_data = "new data";
  std::string obj_id = "some_obj";
  std::string tmp_obj_id = obj_id + ".tmp";
  std::ofstream(tmp_obj_id) << old_data;

  auto put_future =
      service1.AsyncPut((void*)new_data.c_str(), new_data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);
  VerifyFileContentAndRemove(obj_id, new_data);
  EXPECT_FALSE(std::ifstream(tmp_obj_id).good());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutReplacesExistingFile) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string obj_id = "existing_object";
  std::string old_data = "old data";
  std::string new_data = "new data";

  // Create the file with initial content.
  std::ofstream(obj_id) << old_data;

  auto put_future =
      service1.AsyncPut((void*)new_data.c_str(), new_data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);

  // Verify the file was replaced with the new data.
  VerifyFileContentAndRemove(obj_id, new_data);

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, PutFailsInRenameWhenTargetExistAsADirectory) {
  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "some data";
  std::string target_dir = "target_directory";
  mkdir(target_dir.c_str(), 0755);

  std::string obj_id = target_dir;
  auto put_future =
      service1.AsyncPut((void*)data.c_str(), data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);

  try {
    put_future.get();
    FAIL() << "Expected an exception, but none was thrown.";
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(e.what(),
                testing::HasSubstr("Received error from destination"));
  }

  // Cleanup
  remove((obj_id + ".tmp").c_str());  // Remove temporary file if created
  rmdir(target_dir.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetOverwritesEmptyTempFile) {
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "some data here";
  std::string obj_id = "source_obj";
  std::string dest_obj_id = "dest_obj";
  std::string tmp_dest_obj_id = dest_obj_id + ".tmp";

  // Create source file
  std::ofstream(obj_id) << data;
  // Create empty temp file on destination
  std::ofstream(tmp_dest_obj_id).close();

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  auto get_result = get_future.get();
  EXPECT_TRUE(get_result.success);
  VerifyFileContentAndRemove(dest_obj_id, data);
  EXPECT_FALSE(std::ifstream(tmp_dest_obj_id).good());
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetOverwritesNonEmptyTempFile) {
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string old_data = "old data";
  std::string new_data = "new data";
  std::string obj_id = "source_obj";
  std::string dest_obj_id = "dest_obj";
  std::string tmp_dest_obj_id = dest_obj_id + ".tmp";

  // Create source file
  std::ofstream(obj_id) << new_data;
  // Create non-empty temp file on destination
  std::ofstream(tmp_dest_obj_id) << old_data;

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  auto get_result = get_future.get();
  EXPECT_TRUE(get_result.success);
  VerifyFileContentAndRemove(dest_obj_id, new_data);
  EXPECT_FALSE(std::ifstream(tmp_dest_obj_id).good());
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetReplacesExistingFile) {
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string old_data = "old data";
  std::string new_data = "new data";
  std::string obj_id = "source_obj";
  std::string dest_obj_id = "dest_obj";

  // Create source file
  std::ofstream(obj_id) << new_data;
  // Create existing file on destination
  std::ofstream(dest_obj_id) << old_data;

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  auto get_result = get_future.get();
  EXPECT_TRUE(get_result.success);

  // Verify the file was replaced with the new data.
  VerifyFileContentAndRemove(dest_obj_id, new_data);
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, GetFailsInRenameWhenTargetExistAsADirectory) {
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string data = "some data";
  std::string obj_id = "source_obj";
  std::string target_dir = "target_directory_get";
  mkdir(target_dir.c_str(), 0755);

  // Create source file
  std::ofstream(obj_id) << data;

  std::string dest_obj_id = target_dir;
  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);

  try {
    get_future.get();
    FAIL() << "Expected an exception, but none was thrown.";
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(e.what(),
                testing::HasSubstr("Failed to rename temporary file"));
  }

  // Cleanup
  std::remove(obj_id.c_str());
  remove((dest_obj_id + ".tmp").c_str());
  rmdir(target_dir.c_str());

  service1.Shutdown();
  service2.Shutdown();
}

TEST(TransferServiceP2PTest, TimestampsAreRecorded) {
  TestLogSink sink;
  absl::AddLogSink(&sink);

  TransferService service1;
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2;
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  // Perform a Put
  std::string data = "Timestamp test data";
  std::string obj_id = "timestamp_obj_log";
  auto put_future =
      service1.AsyncPut((void*)data.c_str(), data.size(),
                        "127.0.0.1:" + std::to_string(port2), obj_id);
  auto put_result = put_future.get();
  EXPECT_TRUE(put_result.success);
  VerifyFileContentAndRemove(obj_id, data);

  // Perform a Get
  std::string dest_obj_id = "timestamp_obj_get_log";
  std::ofstream out_file(obj_id);
  out_file << data;
  out_file.close();

  auto get_future = service2.AsyncGet(
      obj_id, "127.0.0.1:" + std::to_string(port1), dest_obj_id);
  auto get_result = get_future.get();
  EXPECT_TRUE(get_result.success);
  VerifyFileContentAndRemove(dest_obj_id, data);
  std::remove(obj_id.c_str());

  service1.Shutdown();
  service2.Shutdown();

  absl::RemoveLogSink(&sink);

  int timing_logs_count = 0;
  for (const auto& msg : sink.messages) {
    if (msg.find("timing=") != std::string::npos) {
      timing_logs_count++;
      std::string timing_str = msg.substr(msg.find("timing=") + 7);

      double wait = 0, conn = 0, header = 0, total = 0;
      double data = 0;

      if (timing_str.find("data_sent=") != std::string::npos) {
        int parsed =
            std::sscanf(timing_str.c_str(),
                        "wait_to_be_executed=%lfms, connection_acquired=%lfms, "
                        "header_sent=%lfms, data_sent=%lfms, total=%lfms",
                        &wait, &conn, &header, &data, &total);
        EXPECT_EQ(parsed, 5)
            << "Failed to parse Put/RespondToGet timing: " << timing_str;
      } else {
        int parsed =
            std::sscanf(timing_str.c_str(),
                        "wait_to_be_executed=%lfms, connection_acquired=%lfms, "
                        "header_sent=%lfms, data_received=%lfms, total=%lfms",
                        &wait, &conn, &header, &data, &total);
        EXPECT_EQ(parsed, 5) << "Failed to parse Get timing: " << timing_str;
      }

      EXPECT_GT(wait, 0.0);
      EXPECT_GT(conn, 0.0);
      EXPECT_GT(header, 0.0);
      EXPECT_GT(data, 0.0);
      EXPECT_GT(total, 0.0);
    }
  }
  // We expect 3 tasks: Put, Get, RespondToGet
  EXPECT_EQ(timing_logs_count, 3) << "timing_logs_count: " << timing_logs_count;
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
