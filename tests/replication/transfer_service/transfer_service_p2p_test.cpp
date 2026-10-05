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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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

// ---------------------------------------------------------------------------
// Helpers for driving a TransferService with a raw client socket, used by the
// kGetObj / dest_address (SSRF) hardening tests below.
// ---------------------------------------------------------------------------

// Connects a blocking TCP socket to 127.0.0.1:port. Returns -1 on failure.
// A positive `rcvbuf_bytes` shrinks SO_RCVBUF before connecting, so that a
// peer writing to this socket fills the pipe quickly and blocks in send().
int ConnectToLocalPort(int port, int rcvbuf_bytes = 0) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  if (rcvbuf_bytes > 0) {
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf_bytes, sizeof(rcvbuf_bytes));
  }
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

// Creates a non-blocking listener on 127.0.0.1 with an ephemeral port. Returns
// the listener fd and sets `port` to the bound port, or returns -1 on failure.
int ListenOnLocalEphemeralPort(int* port) {
  int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
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

// Sends `request` on `client_fd` and returns the response header.
ObjInfoHeader SendRequestAndRecvResponse(int client_fd,
                                         const ObjInfoHeader& request) {
  ObjInfoHeader response;
  EXPECT_TRUE(SendAll(client_fd, &request, kHeaderSize).ok());
  EXPECT_TRUE(RecvHeader(client_fd, response).ok());
  return response;
}

// Reads the payload announced by a kRespondToGetObj `response` from `fd` and
// acknowledges it so the responder completes. Returns the payload, or an empty
// string if `response` is not a kRespondToGetObj with a payload.
std::string RecvGetPayloadAndAck(int fd, const ObjInfoHeader& response) {
  if (response.type != MessageType::kRespondToGetObj ||
      response.obj_size <= 0) {
    return "";
  }
  std::string payload(response.obj_size, '\0');
  EXPECT_TRUE(RecvAll(fd, payload.data(), response.obj_size).ok());
  ObjInfoHeader ack;
  ack.type = MessageType::kAck;
  EXPECT_TRUE(SendAll(fd, &ack, kHeaderSize).ok());
  return payload;
}

// Returns true if no connection attempt reaches `listener` within `timeout_ms`.
bool ListenerStaysIdle(int listener, int timeout_ms) {
  pollfd pfd{listener, POLLIN, 0};
  return poll(&pfd, 1, timeout_ms) == 0;
}

// Returns true if the peer closes `fd` (EOF or reset) within `timeout_ms`.
// Any data still arriving before the close is drained and ignored.
bool PeerClosed(int fd, int timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  char drain[256];
  while (true) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
    if (remaining.count() < 0) return false;
    pollfd pfd{fd, POLLIN, 0};
    if (poll(&pfd, 1, static_cast<int>(remaining.count())) <= 0) return false;
    if (recv(fd, drain, sizeof(drain), 0) <= 0) return true;
  }
}

// Runs `service.Shutdown()` on a helper thread and returns true if it completes
// within `timeout`. If it does not, `*blocking_fd` is closed (and set to -1) to
// release whichever worker Shutdown() is stuck behind, so that the thread can
// still be joined and the test fails cleanly instead of hanging.
bool ShutdownCompletesWithin(TransferService& service,
                             std::chrono::milliseconds timeout,
                             int* blocking_fd) {
  std::promise<void> done;
  std::future<void> done_future = done.get_future();
  std::thread shutdown_thread([&service, &done]() {
    service.Shutdown();
    done.set_value();
  });
  const bool completed =
      done_future.wait_for(timeout) == std::future_status::ready;
  if (!completed) {
    close(*blocking_fd);
    *blocking_fd = -1;
  }
  shutdown_thread.join();
  return completed;
}

// Reads a header from `fd` if one starts arriving within `timeout_ms`. Returns
// false on timeout or read failure, so that a stalled or crashed worker fails
// the test instead of hanging it.
bool RecvHeaderWithin(int fd, ObjInfoHeader* header, int timeout_ms) {
  pollfd pfd{fd, POLLIN, 0};
  if (poll(&pfd, 1, timeout_ms) <= 0) return false;
  return RecvHeader(fd, *header).ok();
}

// Returns true if `future` becomes ready within `timeout`.
template <typename T>
bool FutureReadyWithin(const std::future<T>& future,
                       std::chrono::milliseconds timeout) {
  return future.wait_for(timeout) == std::future_status::ready;
}

ObjInfoHeader BuildPutRequest(const std::string& task_id,
                              const std::string& dest_obj_id,
                              ssize_t obj_size) {
  ObjInfoHeader request;
  request.type = MessageType::kPutObj;
  snprintf(request.task_id, sizeof(request.task_id), "%s", task_id.c_str());
  snprintf(request.dest_obj_id, sizeof(request.dest_obj_id), "%s",
           dest_obj_id.c_str());
  request.obj_size = obj_size;
  return request;
}

// Stands in for a misbehaving peer: accepts one connection, reads one kGetObj
// request and answers it with whatever `respond` writes to the socket. Every
// wait is bounded so the helper thread always terminates.
class FakeGetResponder {
 public:
  explicit FakeGetResponder(std::function<void(int fd)> respond)
      : respond_(std::move(respond)) {
    listener_fd_ = ListenOnLocalEphemeralPort(&port_);
    thread_ = std::thread([this]() { Serve(); });
  }

  ~FakeGetResponder() {
    thread_.join();
    if (listener_fd_ >= 0) close(listener_fd_);
  }

  std::string address() const { return "127.0.0.1:" + std::to_string(port_); }
  bool received_request() const { return received_request_; }

 private:
  void Serve() {
    if (listener_fd_ < 0) return;
    pollfd pfd{listener_fd_, POLLIN, 0};
    if (poll(&pfd, 1, /*timeout_ms=*/5000) <= 0) return;
    int client_fd = accept(listener_fd_, nullptr, nullptr);
    if (client_fd < 0) return;
    ObjInfoHeader request;
    if (RecvHeaderWithin(client_fd, &request, /*timeout_ms=*/5000) &&
        request.type == MessageType::kGetObj) {
      received_request_ = true;
      respond_(client_fd);
    }
    close(client_fd);
  }

  std::function<void(int fd)> respond_;
  int listener_fd_ = -1;
  int port_ = 0;
  std::atomic<bool> received_request_{false};
  std::thread thread_;
};

// Verifies that handling a kGetObj request responds over the existing client
// socket and never initiates an outbound connection to header.dest_address.
TEST(TransferServiceP2PTest, GetObjDoesNotConnectToDestAddress) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  std::string expected_data = "ssrf_test_payload";
  std::string source_obj_id = "ssrf_source_obj";
  std::ofstream(source_obj_id) << expected_data;

  int target_listener_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
  ASSERT_GE(target_listener_fd, 0);
  sockaddr_in target_addr{};
  target_addr.sin_family = AF_INET;
  inet_pton(AF_INET, "127.0.0.1", &target_addr.sin_addr);
  target_addr.sin_port = 0;
  ASSERT_EQ(bind(target_listener_fd, reinterpret_cast<sockaddr*>(&target_addr),
                 sizeof(target_addr)),
            0);
  socklen_t target_len = sizeof(target_addr);
  ASSERT_EQ(getsockname(target_listener_fd,
                        reinterpret_cast<sockaddr*>(&target_addr), &target_len),
            0);
  int target_port = ntohs(target_addr.sin_port);
  ASSERT_EQ(listen(target_listener_fd, 1), 0);

  int client_fd = socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(client_fd, 0);
  sockaddr_in serv_addr{};
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port1);
  inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
  ASSERT_EQ(connect(client_fd, reinterpret_cast<sockaddr*>(&serv_addr),
                    sizeof(serv_addr)),
            0);

  // When
  ObjInfoHeader req_header;
  req_header.type = MessageType::kGetObj;
  snprintf(req_header.task_id, sizeof(req_header.task_id), "test_task_id");
  snprintf(req_header.source_obj_id, sizeof(req_header.source_obj_id), "%s",
           source_obj_id.c_str());
  snprintf(req_header.dest_obj_id, sizeof(req_header.dest_obj_id),
           "ssrf_dest_obj");
  snprintf(req_header.dest_address, sizeof(req_header.dest_address),
           "127.0.0.1:%d", target_port);
  ASSERT_TRUE(SendAll(client_fd, &req_header, kHeaderSize).ok());

  ObjInfoHeader resp_header;
  ASSERT_TRUE(RecvHeader(client_fd, resp_header).ok());
  std::string actual_data(resp_header.obj_size, '\0');
  ASSERT_TRUE(
      RecvAll(client_fd, actual_data.data(), resp_header.obj_size).ok());

  ObjInfoHeader ack_header;
  ack_header.type = MessageType::kAck;
  ASSERT_TRUE(SendAll(client_fd, &ack_header, kHeaderSize).ok());

  // Then
  EXPECT_EQ(resp_header.type, MessageType::kRespondToGetObj);
  EXPECT_EQ(actual_data, expected_data);
  EXPECT_TRUE(ListenerStaysIdle(target_listener_fd, /*timeout_ms=*/200));

  close(client_fd);
  close(target_listener_fd);
  std::remove(source_obj_id.c_str());
  service1.Shutdown();
}

class SpoofedDestAddressTest : public ::testing::TestWithParam<std::string> {};

// Whatever dest_address a kGetObj request carries (other hosts, metadata IP,
// garbage, invalid or missing port), the object is streamed back over the
// request's own socket and no outbound connection is ever attempted.
TEST_P(SpoofedDestAddressTest, GetResponseGoesBackOnRequestSocket) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string expected_data = "spoofed_dest_payload";
  const std::string source_obj_id = "spoofed_dest_source_obj";
  WriteTestFile(source_obj_id, expected_data);
  int target_port = 0;
  int target_listener_fd = ListenOnLocalEphemeralPort(&target_port);
  ASSERT_GE(target_listener_fd, 0);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  // "{port}" stands in for a real, listening port so that a regression to
  // connecting back would be observable on target_listener_fd.
  std::string dest_address = GetParam();
  const size_t placeholder = dest_address.find("{port}");
  if (placeholder != std::string::npos) {
    dest_address.replace(placeholder, 6, std::to_string(target_port));
  }

  // When
  ObjInfoHeader actual_response = SendRequestAndRecvResponse(
      client_fd, BuildGetRequest("spoofed_task", source_obj_id,
                                 "spoofed_dest_obj", dest_address));
  std::string actual_data = RecvGetPayloadAndAck(client_fd, actual_response);

  // Then
  EXPECT_EQ(actual_response.type, MessageType::kRespondToGetObj);
  EXPECT_STREQ(actual_response.task_id, "spoofed_task");
  EXPECT_EQ(actual_data, expected_data);
  EXPECT_TRUE(ListenerStaysIdle(target_listener_fd, /*timeout_ms=*/200));

  close(client_fd);
  close(target_listener_fd);
  std::remove(source_obj_id.c_str());
  service1.Shutdown();
}

INSTANTIATE_TEST_SUITE_P(
    TransferServiceP2PTest, SpoofedDestAddressTest,
    ::testing::Values("127.0.0.1:{port}",        // the requester itself
                      "203.0.113.1:{port}",      // TEST-NET-3
                      "10.0.0.1:{port}",         // private
                      "169.254.169.254:{port}",  // metadata server
                      "0.0.0.0:{port}", "255.255.255.255:{port}",
                      "not-a-host:{port}", "127.0.0.1:0", "127.0.0.1:65536",
                      "127.0.0.1:abc", ":", ""));

// Address fields that fill their whole buffer without a NUL terminator are
// handled safely (no over-read) and the request is served like any other.
TEST(TransferServiceP2PTest, GetRequestWithUnterminatedAddressFieldsIsServed) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string expected_data = "unterminated_payload";
  const std::string source_obj_id = "unterminated_source_obj";
  WriteTestFile(source_obj_id, expected_data);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader request = BuildGetRequest("unterminated_task", source_obj_id,
                                          "unterminated_dest_obj", "");
  std::memset(request.dest_address, 'A', sizeof(request.dest_address));
  std::memset(request.source_address, 'B', sizeof(request.source_address));

  // When
  ObjInfoHeader actual_response =
      SendRequestAndRecvResponse(client_fd, request);
  std::string actual_data = RecvGetPayloadAndAck(client_fd, actual_response);

  // Then
  EXPECT_EQ(actual_response.type, MessageType::kRespondToGetObj);
  EXPECT_STREQ(actual_response.task_id, "unterminated_task");
  EXPECT_EQ(actual_data, expected_data);

  close(client_fd);
  std::remove(source_obj_id.c_str());
  service1.Shutdown();
}

// An object that exists but cannot be opened (empty file, directory) yields a
// kError on the request socket rather than an exception that would orphan the
// connection and hang the requester, and the socket stays usable afterwards.
TEST(TransferServiceP2PTest,
     GetRequestForUnopenableObjectReturnsErrorOnSameSocket) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string empty_obj_id = "unopenable_empty_obj";
  const std::string dir_obj_id = "unopenable_dir_obj";
  const std::string valid_obj_id = "unopenable_valid_obj";
  const std::string expected_data = "payload_after_errors";
  WriteTestFile(empty_obj_id, "");
  std::filesystem::create_directory(dir_obj_id);
  WriteTestFile(valid_obj_id, expected_data);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When
  ObjInfoHeader actual_empty_response = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("empty_task", empty_obj_id, "unopenable_empty_dest", ""));
  ObjInfoHeader actual_dir_response = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("dir_task", dir_obj_id, "unopenable_dir_dest", ""));
  ObjInfoHeader actual_valid_response = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("valid_task", valid_obj_id, "unopenable_valid_dest", ""));
  std::string actual_data =
      RecvGetPayloadAndAck(client_fd, actual_valid_response);

  // Then
  EXPECT_EQ(actual_empty_response.type, MessageType::kError);
  EXPECT_STREQ(actual_empty_response.task_id, "empty_task");
  EXPECT_EQ(actual_dir_response.type, MessageType::kError);
  EXPECT_STREQ(actual_dir_response.task_id, "dir_task");
  EXPECT_EQ(actual_valid_response.type, MessageType::kRespondToGetObj);
  EXPECT_STREQ(actual_valid_response.task_id, "valid_task");
  EXPECT_EQ(actual_data, expected_data);

  close(client_fd);
  std::remove(empty_obj_id.c_str());
  std::filesystem::remove(dir_obj_id);
  std::remove(valid_obj_id.c_str());
  service1.Shutdown();
}

// An inbound kRespondToGetObj that no GetTask is waiting for must not act as
// an arbitrary file write: the connection is closed and nothing is written.
TEST(TransferServiceP2PTest,
     UnsolicitedRespondToGetObjClosesConnectionWithoutWritingFile) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string dest_obj_id = "unsolicited_dest_obj";
  std::remove(dest_obj_id.c_str());
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  const std::string payload = "pwnd";
  ObjInfoHeader header;
  header.type = MessageType::kRespondToGetObj;
  snprintf(header.task_id, sizeof(header.task_id), "unsolicited_task");
  snprintf(header.dest_obj_id, sizeof(header.dest_obj_id), "%s",
           dest_obj_id.c_str());
  header.obj_size = payload.size();

  // When
  ASSERT_TRUE(SendAll(client_fd, &header, kHeaderSize).ok());
  // The service may already have closed the socket, so this send may fail.
  send(client_fd, payload.data(), payload.size(), MSG_NOSIGNAL);

  // Then
  EXPECT_TRUE(PeerClosed(client_fd, /*timeout_ms=*/5000));
  EXPECT_FALSE(std::filesystem::exists(dest_obj_id));
  EXPECT_FALSE(std::filesystem::exists(dest_obj_id + ".tmp"));

  close(client_fd);
  service1.Shutdown();
}

// A header with an unknown message type leaves the stream desynchronised, so
// the service closes the connection instead of guessing.
TEST(TransferServiceP2PTest, UnknownMessageTypeClosesConnection) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader header;
  header.type = static_cast<MessageType>(42);
  snprintf(header.task_id, sizeof(header.task_id), "unknown_type_task");

  // When
  ASSERT_TRUE(SendAll(client_fd, &header, kHeaderSize).ok());

  // Then
  EXPECT_TRUE(PeerClosed(client_fd, /*timeout_ms=*/5000));

  close(client_fd);
  service1.Shutdown();
}

// Shutdown() must neither crash on the promise-less RespondToGetTask nor hang
// behind a responder that is blocked waiting for the requester's final ACK on
// an accepted socket.
TEST(TransferServiceP2PTest, ShutdownWhileResponderWaitsForAckIsSafe) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string expected_data = "shutdown_no_ack_payload";
  const std::string source_obj_id = "shutdown_no_ack_source_obj";
  WriteTestFile(source_obj_id, expected_data);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader actual_response = SendRequestAndRecvResponse(
      client_fd, BuildGetRequest("no_ack_task", source_obj_id,
                                 "shutdown_no_ack_dest_obj", ""));
  ASSERT_EQ(actual_response.type, MessageType::kRespondToGetObj);
  std::string actual_data(actual_response.obj_size, '\0');
  ASSERT_TRUE(
      RecvAll(client_fd, actual_data.data(), actual_response.obj_size).ok());
  // Deliberately no ACK: the responder is now blocked in RecvHeader.

  // When
  const bool actual_completed =
      ShutdownCompletesWithin(service1, std::chrono::seconds(10), &client_fd);

  // Then
  EXPECT_TRUE(actual_completed)
      << "Shutdown() hung behind a responder waiting for an ACK";
  EXPECT_EQ(actual_data, expected_data);
  if (client_fd >= 0) {
    EXPECT_TRUE(PeerClosed(client_fd, /*timeout_ms=*/1000));
    close(client_fd);
  }
  TransferService service2("127.0.0.1");
  EXPECT_GT(service2.Initialize(), 0);
  service2.Shutdown();
  std::remove(source_obj_id.c_str());
}

// Shutdown() must not hang behind a responder whose peer has stopped reading,
// i.e. a worker blocked in send() on an accepted socket with a full buffer.
TEST(TransferServiceP2PTest, ShutdownWhilePeerIsNotReadingGetResponseIsSafe) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  // Large enough to exceed any socket buffer autotuning on loopback.
  const std::string large_data(16 * 1024 * 1024, 'L');
  const std::string source_obj_id = "shutdown_not_reading_source_obj";
  WriteTestFile(source_obj_id, large_data);
  int client_fd = ConnectToLocalPort(port1, /*rcvbuf_bytes=*/4096);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader request = BuildGetRequest("not_reading_task", source_obj_id,
                                          "shutdown_not_reading_dest_obj", "");
  ASSERT_TRUE(SendAll(client_fd, &request, kHeaderSize).ok());
  // Wait until the response starts arriving, then let the responder fill the
  // socket buffers and block, without ever reading from client_fd.
  pollfd pfd{client_fd, POLLIN, 0};
  ASSERT_GT(poll(&pfd, 1, /*timeout_ms=*/5000), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  // When
  const bool actual_completed =
      ShutdownCompletesWithin(service1, std::chrono::seconds(10), &client_fd);

  // Then
  EXPECT_TRUE(actual_completed)
      << "Shutdown() hung behind a responder blocked in send()";
  if (client_fd >= 0) close(client_fd);
  std::remove(source_obj_id.c_str());
}

// Happy path through the public API: because the response travels back over
// the request's own connection, a Get succeeds even when the requesting
// service advertises a local address that is not reachable.
TEST(TransferServiceP2PTest, GetSucceedsWhenRequesterAdvertisesUnreachableIp) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);
  TransferService service2("203.0.113.5");
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);
  const std::string expected_data = "reachable after all";
  const std::string source_obj_id = "unreachable_requester_source_obj";
  const std::string dest_obj_id = "unreachable_requester_dest_obj";
  WriteTestFile(source_obj_id, expected_data);

  // When
  TransferResult actual_result =
      service2
          .AsyncGet(source_obj_id, "127.0.0.1:" + std::to_string(port1),
                    dest_obj_id)
          .get();

  // Then
  EXPECT_TRUE(actual_result.success);
  VerifyFileContentAndRemove(dest_obj_id, expected_data);

  std::remove(source_obj_id.c_str());
  service1.Shutdown();
  service2.Shutdown();
}

// Several kGetObj requests issued back-to-back on one accepted socket are each
// answered on that socket, in order, with the matching task_id and payload.
TEST(TransferServiceP2PTest, SequentialGetRequestsOnOneSocketAreServedInOrder) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const int kNumRequests = 3;
  std::vector<std::string> source_obj_ids;
  std::vector<std::string> expected_payloads;
  for (int i = 0; i < kNumRequests; ++i) {
    source_obj_ids.push_back("sequential_source_obj_" + std::to_string(i));
    expected_payloads.push_back("sequential_payload_" + std::to_string(i) +
                                std::string(4096 * (i + 1), 'a' + i));
    WriteTestFile(source_obj_ids[i], expected_payloads[i]);
  }
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When
  std::vector<ObjInfoHeader> actual_responses;
  std::vector<std::string> actual_payloads;
  for (int i = 0; i < kNumRequests; ++i) {
    actual_responses.push_back(SendRequestAndRecvResponse(
        client_fd,
        BuildGetRequest("sequential_task_" + std::to_string(i),
                        source_obj_ids[i], "sequential_dest_obj", "")));
    actual_payloads.push_back(
        RecvGetPayloadAndAck(client_fd, actual_responses.back()));
  }

  // Then
  for (int i = 0; i < kNumRequests; ++i) {
    EXPECT_EQ(actual_responses[i].type, MessageType::kRespondToGetObj);
    EXPECT_EQ(std::string(actual_responses[i].task_id),
              "sequential_task_" + std::to_string(i));
    EXPECT_EQ(actual_payloads[i], expected_payloads[i]);
  }

  close(client_fd);
  for (const auto& source_obj_id : source_obj_ids) {
    std::remove(source_obj_id.c_str());
  }
  service1.Shutdown();
}

// Because a kGetObj response now occupies an inbound worker for the whole
// transfer, more simultaneous requests than inbound workers must still all be
// served (queued behind each other) rather than dropped or deadlocked.
TEST(TransferServiceP2PTest, ConcurrentGetRequestsBeyondThreadCountAreServed) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/2,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const int kNumClients = 8;
  std::vector<std::string> source_obj_ids(kNumClients);
  std::vector<std::string> expected_payloads(kNumClients);
  for (int i = 0; i < kNumClients; ++i) {
    source_obj_ids[i] = "oversubscribed_source_obj_" + std::to_string(i);
    expected_payloads[i] = "oversubscribed_payload_" + std::to_string(i) +
                           std::string(64 * 1024, 'A' + i);
    WriteTestFile(source_obj_ids[i], expected_payloads[i]);
  }
  std::vector<MessageType> actual_types(kNumClients, MessageType::kError);
  std::vector<std::string> actual_payloads(kNumClients);
  // Not vector<bool>: its packed elements cannot be written concurrently.
  std::vector<int> actual_responded(kNumClients, 0);

  // When
  std::vector<std::thread> clients;
  for (int i = 0; i < kNumClients; ++i) {
    clients.emplace_back([&, i]() {
      int client_fd = ConnectToLocalPort(port1);
      if (client_fd < 0) return;
      ObjInfoHeader request =
          BuildGetRequest("oversubscribed_task_" + std::to_string(i),
                          source_obj_ids[i], "oversubscribed_dest_obj", "");
      ObjInfoHeader response;
      if (SendAll(client_fd, &request, kHeaderSize).ok() &&
          RecvHeaderWithin(client_fd, &response, /*timeout_ms=*/10000)) {
        actual_responded[i] = 1;
        actual_types[i] = response.type;
        actual_payloads[i] = RecvGetPayloadAndAck(client_fd, response);
      }
      close(client_fd);
    });
  }
  for (auto& client : clients) client.join();

  // Then
  for (int i = 0; i < kNumClients; ++i) {
    EXPECT_TRUE(actual_responded[i]) << "client " << i << " got no response";
    EXPECT_EQ(actual_types[i], MessageType::kRespondToGetObj) << "client " << i;
    EXPECT_EQ(actual_payloads[i], expected_payloads[i]) << "client " << i;
  }

  for (const auto& source_obj_id : source_obj_ids) {
    std::remove(source_obj_id.c_str());
  }
  service1.Shutdown();
}

// A requester that disconnects while its response is still being streamed must
// only fail its own request: with a single inbound worker, the next client's
// request proves the worker was released promptly and the service is healthy.
TEST(TransferServiceP2PTest, ResponderSurvivesClientDisconnectMidResponse) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/1,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  // Large enough to exceed any socket buffer autotuning on loopback.
  const std::string large_data(16 * 1024 * 1024, 'L');
  const std::string large_obj_id = "mid_response_large_source_obj";
  const std::string expected_data = "served after the disconnect";
  const std::string small_obj_id = "mid_response_small_source_obj";
  WriteTestFile(large_obj_id, large_data);
  WriteTestFile(small_obj_id, expected_data);
  int leaving_fd = ConnectToLocalPort(port1, /*rcvbuf_bytes=*/4096);
  ASSERT_GE(leaving_fd, 0);
  ObjInfoHeader large_request = BuildGetRequest(
      "mid_response_leaving_task", large_obj_id, "mid_response_dest_obj", "");
  ASSERT_TRUE(SendAll(leaving_fd, &large_request, kHeaderSize).ok());
  // Let the response start flowing and the responder block in send() on the
  // full socket buffers, then walk away with most of the payload unread.
  pollfd pfd{leaving_fd, POLLIN, 0};
  ASSERT_GT(poll(&pfd, 1, /*timeout_ms=*/5000), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // When
  close(leaving_fd);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader request = BuildGetRequest(
      "mid_response_next_task", small_obj_id, "mid_response_dest_obj", "");
  ASSERT_TRUE(SendAll(client_fd, &request, kHeaderSize).ok());
  ObjInfoHeader actual_response;
  const bool actual_responded =
      RecvHeaderWithin(client_fd, &actual_response, /*timeout_ms=*/10000);

  // Then
  ASSERT_TRUE(actual_responded)
      << "the single inbound worker is still stuck on the departed client";
  EXPECT_EQ(actual_response.type, MessageType::kRespondToGetObj);
  EXPECT_EQ(RecvGetPayloadAndAck(client_fd, actual_response), expected_data);

  close(client_fd);
  std::remove(large_obj_id.c_str());
  std::remove(small_obj_id.c_str());
  service1.Shutdown();
}

// A requester that reads the whole response but closes instead of sending the
// final ACK must not pin the inbound worker: the next client is still served.
TEST(TransferServiceP2PTest, ResponderSurvivesClientClosingInsteadOfAck) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/1,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string expected_data = "no_ack_then_next_payload";
  const std::string source_obj_id = "no_ack_next_source_obj";
  WriteTestFile(source_obj_id, expected_data);
  int leaving_fd = ConnectToLocalPort(port1);
  ASSERT_GE(leaving_fd, 0);
  ObjInfoHeader first_response = SendRequestAndRecvResponse(
      leaving_fd,
      BuildGetRequest("no_ack_leaving_task", source_obj_id, "no_ack_dest", ""));
  ASSERT_EQ(first_response.type, MessageType::kRespondToGetObj);
  std::string first_payload(first_response.obj_size, '\0');
  ASSERT_TRUE(
      RecvAll(leaving_fd, first_payload.data(), first_response.obj_size).ok());

  // When
  close(leaving_fd);  // Deliberately no ACK.
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader request =
      BuildGetRequest("no_ack_next_task", source_obj_id, "no_ack_dest", "");
  ASSERT_TRUE(SendAll(client_fd, &request, kHeaderSize).ok());
  ObjInfoHeader actual_response;
  const bool actual_responded =
      RecvHeaderWithin(client_fd, &actual_response, /*timeout_ms=*/10000);

  // Then
  EXPECT_EQ(first_payload, expected_data);
  ASSERT_TRUE(actual_responded)
      << "the single inbound worker is still waiting for the departed ACK";
  EXPECT_EQ(actual_response.type, MessageType::kRespondToGetObj);
  EXPECT_EQ(RecvGetPayloadAndAck(client_fd, actual_response), expected_data);

  close(client_fd);
  std::remove(source_obj_id.c_str());
  service1.Shutdown();
}

class InvalidGetRequestTest : public ::testing::TestWithParam<std::string> {};

// A kGetObj naming an object that cannot be served (empty id, unknown id, or an
// id that fills its whole field without a terminator) is answered with a kError
// carrying the request's task_id, and the same socket then serves a valid GET.
TEST_P(InvalidGetRequestTest, RepliesWithErrorAndKeepsSocketUsable) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string expected_data = "valid_after_invalid_request";
  const std::string valid_obj_id = "invalid_request_valid_source_obj";
  WriteTestFile(valid_obj_id, expected_data);
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);
  ObjInfoHeader invalid_request =
      BuildGetRequest("invalid_task", GetParam(), "invalid_dest_obj", "");
  if (GetParam() == "{unterminated}") {
    std::memset(invalid_request.source_obj_id, 'A',
                sizeof(invalid_request.source_obj_id));
  }

  // When
  ObjInfoHeader actual_error_response =
      SendRequestAndRecvResponse(client_fd, invalid_request);
  ObjInfoHeader actual_valid_response = SendRequestAndRecvResponse(
      client_fd,
      BuildGetRequest("valid_task", valid_obj_id, "invalid_request_dest", ""));
  std::string actual_data =
      RecvGetPayloadAndAck(client_fd, actual_valid_response);

  // Then
  EXPECT_EQ(actual_error_response.type, MessageType::kError);
  EXPECT_STREQ(actual_error_response.task_id, "invalid_task");
  EXPECT_EQ(actual_valid_response.type, MessageType::kRespondToGetObj);
  EXPECT_STREQ(actual_valid_response.task_id, "valid_task");
  EXPECT_EQ(actual_data, expected_data);

  close(client_fd);
  std::remove(valid_obj_id.c_str());
  service1.Shutdown();
}

INSTANTIATE_TEST_SUITE_P(TransferServiceP2PTest, InvalidGetRequestTest,
                         ::testing::Values("", "non_existent_source_obj",
                                           "{unterminated}"));

// When the receiver cannot create the destination of an inbound kPutObj, it
// drains the payload the sender is already streaming and answers kError, so the
// connection stays aligned and the next request on it is served normally.
TEST(TransferServiceP2PTest,
     PutToUncreatableDestinationRepliesWithErrorAndKeepsSocketUsable) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  const std::string undeliverable_payload(256 * 1024, 'U');
  // A regular file as a path component makes the destination uncreatable
  // (missing directories alone would simply be created).
  const std::string not_a_dir = "put_not_a_dir";
  WriteTestFile(not_a_dir, "regular file");
  const std::string uncreatable_dest = not_a_dir + "/uncreatable_dest_obj";
  const std::string expected_data = "delivered_after_undeliverable_put";
  const std::string dest_obj_id = "uncreatable_then_valid_dest_obj";
  std::remove(dest_obj_id.c_str());
  int client_fd = ConnectToLocalPort(port1);
  ASSERT_GE(client_fd, 0);

  // When
  ObjInfoHeader uncreatable_request = BuildPutRequest(
      "uncreatable_task", uncreatable_dest, undeliverable_payload.size());
  ASSERT_TRUE(SendAll(client_fd, &uncreatable_request, kHeaderSize).ok());
  ASSERT_TRUE(SendAll(client_fd, undeliverable_payload.data(),
                      undeliverable_payload.size())
                  .ok());
  ObjInfoHeader actual_error_response;
  const bool actual_error_responded =
      RecvHeaderWithin(client_fd, &actual_error_response, /*timeout_ms=*/10000);
  ObjInfoHeader valid_request =
      BuildPutRequest("valid_put_task", dest_obj_id, expected_data.size());
  ASSERT_TRUE(SendAll(client_fd, &valid_request, kHeaderSize).ok());
  ASSERT_TRUE(
      SendAll(client_fd, expected_data.data(), expected_data.size()).ok());
  ObjInfoHeader actual_ack_response;
  const bool actual_ack_responded =
      RecvHeaderWithin(client_fd, &actual_ack_response, /*timeout_ms=*/10000);

  // Then
  ASSERT_TRUE(actual_error_responded);
  EXPECT_EQ(actual_error_response.type, MessageType::kError);
  EXPECT_STREQ(actual_error_response.task_id, "uncreatable_task");
  ASSERT_TRUE(actual_ack_responded);
  EXPECT_EQ(actual_ack_response.type, MessageType::kAck);
  EXPECT_TRUE(std::filesystem::is_regular_file(not_a_dir));
  VerifyFileContentAndRemove(dest_obj_id, expected_data);

  close(client_fd);
  std::remove(not_a_dir.c_str());
  service1.Shutdown();
}

// Through the public API, an AsyncPut whose destination cannot be created on
// the peer fails with an error (not a hang), and the single pooled connection
// it used is still good for the next AsyncPut.
TEST(TransferServiceP2PTest,
     PutFailsCleanlyAndReusesConnectionWhenDestinationCannotBeCreated) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port2, 0);
  const std::string peer1_addr = "127.0.0.1:" + std::to_string(port1);
  std::string undeliverable_payload(256 * 1024, 'U');
  std::string expected_data = "put_after_uncreatable_destination";
  const std::string not_a_dir = "api_put_not_a_dir";
  WriteTestFile(not_a_dir, "regular file");
  const std::string dest_obj_id = "api_put_uncreatable_then_valid_dest_obj";

  // When
  std::future<TransferResult> failed_future = service2.AsyncPut(
      undeliverable_payload.data(), undeliverable_payload.size(), peer1_addr,
      not_a_dir + "/uncreatable_dest_obj");
  const bool actual_failed_ready =
      FutureReadyWithin(failed_future, std::chrono::seconds(10));
  std::future<TransferResult> valid_future = service2.AsyncPut(
      expected_data.data(), expected_data.size(), peer1_addr, dest_obj_id);
  const bool actual_valid_ready =
      FutureReadyWithin(valid_future, std::chrono::seconds(10));

  // Then
  ASSERT_TRUE(actual_failed_ready) << "AsyncPut hung instead of failing";
  EXPECT_THROW(failed_future.get(), std::runtime_error);
  ASSERT_TRUE(actual_valid_ready) << "pooled connection unusable after error";
  EXPECT_TRUE(valid_future.get().success);
  EXPECT_TRUE(std::filesystem::is_regular_file(not_a_dir));
  VerifyFileContentAndRemove(dest_obj_id, expected_data);

  std::remove(not_a_dir.c_str());
  service1.Shutdown();
  service2.Shutdown();
}

// Through the public API, an AsyncGet whose local destination cannot be created
// fails with an error (not a hang), the responder is told so it does not wait
// for an ACK, and the single pooled connection is still good for the next Get.
TEST(TransferServiceP2PTest,
     GetFailsCleanlyAndReusesConnectionWhenDestinationCannotBeCreated) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);
  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize(/*listen_port=*/0, /*threads=*/4,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port2, 0);
  const std::string peer1_addr = "127.0.0.1:" + std::to_string(port1);
  const std::string large_obj_id = "api_get_uncreatable_large_source_obj";
  const std::string source_obj_id = "api_get_uncreatable_valid_source_obj";
  const std::string not_a_dir = "api_get_not_a_dir";
  const std::string dest_obj_id = "api_get_uncreatable_then_valid_dest_obj";
  const std::string expected_data = "get_after_uncreatable_destination";
  WriteTestFile(large_obj_id, std::string(256 * 1024, 'U'));
  WriteTestFile(source_obj_id, expected_data);
  WriteTestFile(not_a_dir, "regular file");

  // When
  std::future<TransferResult> failed_future = service2.AsyncGet(
      large_obj_id, peer1_addr, not_a_dir + "/uncreatable_dest_obj");
  const bool actual_failed_ready =
      FutureReadyWithin(failed_future, std::chrono::seconds(10));
  std::future<TransferResult> valid_future =
      service2.AsyncGet(source_obj_id, peer1_addr, dest_obj_id);
  const bool actual_valid_ready =
      FutureReadyWithin(valid_future, std::chrono::seconds(10));

  // Then
  ASSERT_TRUE(actual_failed_ready) << "AsyncGet hung instead of failing";
  EXPECT_THROW(failed_future.get(), std::runtime_error);
  ASSERT_TRUE(actual_valid_ready) << "pooled connection unusable after error";
  EXPECT_TRUE(valid_future.get().success);
  EXPECT_TRUE(std::filesystem::is_regular_file(not_a_dir));
  VerifyFileContentAndRemove(dest_obj_id, expected_data);

  std::remove(large_obj_id.c_str());
  std::remove(source_obj_id.c_str());
  std::remove(not_a_dir.c_str());
  service1.Shutdown();
  service2.Shutdown();
}

class MalformedGetResponseTest : public ::testing::TestWithParam<std::string> {
};

// Whatever a misbehaving peer sends back for a GET (zero-size response, wrong
// message type, a bare ACK, a payload shorter than announced, or nothing at
// all), the requester's future resolves with an error instead of hanging.
TEST_P(MalformedGetResponseTest, RequesterFutureFailsInsteadOfHanging) {
  // Given
  const std::string scenario = GetParam();
  const std::string dest_obj_id = "malformed_response_dest_obj";
  FakeGetResponder responder([&scenario](int fd) {
    ObjInfoHeader response;
    if (scenario == "close_without_response") return;
    if (scenario == "zero_size_response") {
      response.type = MessageType::kRespondToGetObj;
      response.obj_size = 0;
    } else if (scenario == "wrong_type_response") {
      response.type = MessageType::kPutObj;
    } else if (scenario == "ack_response") {
      response.type = MessageType::kAck;
    } else if (scenario == "truncated_payload") {
      response.type = MessageType::kRespondToGetObj;
      response.obj_size = 1024;
    }
    snprintf(response.task_id, sizeof(response.task_id), "ignored");
    SendAll(fd, &response, kHeaderSize).IgnoreError();
    if (scenario == "truncated_payload") {
      const std::string partial(16, 'p');
      SendAll(fd, partial.data(), partial.size()).IgnoreError();
    }
  });
  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize(/*listen_port=*/0, /*threads=*/2,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port2, 0);

  // When
  std::future<TransferResult> actual_future =
      service2.AsyncGet("any_source_obj", responder.address(), dest_obj_id);
  const bool actual_ready =
      FutureReadyWithin(actual_future, std::chrono::seconds(10));

  // Then
  ASSERT_TRUE(actual_ready) << "AsyncGet hung on scenario " << scenario;
  EXPECT_THROW(actual_future.get(), std::runtime_error);
  EXPECT_TRUE(responder.received_request());
  EXPECT_FALSE(std::filesystem::exists(dest_obj_id));

  std::remove((dest_obj_id + ".tmp").c_str());
  service2.Shutdown();
}

INSTANTIATE_TEST_SUITE_P(TransferServiceP2PTest, MalformedGetResponseTest,
                         ::testing::Values("zero_size_response",
                                           "wrong_type_response",
                                           "ack_response", "truncated_payload",
                                           "close_without_response"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return info.param;
                         });

// A peer that answers a GET correctly but names a different dest_obj_id (and
// task_id) in its response must not redirect where the payload is written nor
// which task it completes: the file lands at the requester's own destination.
TEST(TransferServiceP2PTest, GetWritesToLocalDestinationNotToPeerNamedPath) {
  // Given
  const std::string expected_data = "payload written where the caller asked";
  const std::string dest_obj_id = "peer_redirect_requested_dest_obj";
  const std::string redirected_obj_id = "peer_redirect_peer_named_obj";
  std::remove(dest_obj_id.c_str());
  std::remove(redirected_obj_id.c_str());
  std::promise<MessageType> ack_promise;
  std::future<MessageType> ack_future = ack_promise.get_future();
  FakeGetResponder responder([&](int fd) {
    ObjInfoHeader response;
    response.type = MessageType::kRespondToGetObj;
    response.obj_size = expected_data.size();
    snprintf(response.task_id, sizeof(response.task_id), "peer_chosen_task");
    snprintf(response.dest_obj_id, sizeof(response.dest_obj_id), "%s",
             redirected_obj_id.c_str());
    SendAll(fd, &response, kHeaderSize).IgnoreError();
    SendAll(fd, expected_data.data(), expected_data.size()).IgnoreError();
    ObjInfoHeader ack;
    ack_promise.set_value(RecvHeaderWithin(fd, &ack, /*timeout_ms=*/5000)
                              ? ack.type
                              : MessageType::kError);
  });
  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize(/*listen_port=*/0, /*threads=*/2,
                                  /*conn_pool_per_peer=*/1);
  ASSERT_GT(port2, 0);

  // When
  std::future<TransferResult> actual_future =
      service2.AsyncGet("any_source_obj", responder.address(), dest_obj_id);
  const bool actual_ready =
      FutureReadyWithin(actual_future, std::chrono::seconds(10));

  // Then
  ASSERT_TRUE(actual_ready) << "AsyncGet hung on a redirecting responder";
  EXPECT_TRUE(actual_future.get().success);
  ASSERT_TRUE(FutureReadyWithin(ack_future, std::chrono::seconds(5)));
  EXPECT_EQ(ack_future.get(), MessageType::kAck);
  VerifyFileContentAndRemove(dest_obj_id, expected_data);
  EXPECT_FALSE(std::filesystem::exists(redirected_obj_id));
  EXPECT_FALSE(std::filesystem::exists(redirected_obj_id + ".tmp"));

  service2.Shutdown();
}

// Verifies that a single pooled connection (conn_pool_per_peer=1) is cleanly
// reused across sequential AsyncGet calls, including after a kError response.
TEST(TransferServiceP2PTest,
     GetReusesSinglePooledConnectionAfterSuccessAndError) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize(0, /*threads=*/4, /*conn_pool_per_peer=*/1);
  ASSERT_GT(port1, 0);

  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize(0, /*threads=*/4, /*conn_pool_per_peer=*/1);
  ASSERT_GT(port2, 0);

  std::string peer1_addr = "127.0.0.1:" + std::to_string(port1);
  std::string expected_data_1 = "first_payload_over_single_conn";
  std::string expected_data_2 = "second_payload_after_error_over_single_conn";
  std::string source_obj_1 = "single_conn_source_1";
  std::string source_obj_2 = "single_conn_source_2";
  std::string dest_obj_1 = "single_conn_dest_1";
  std::string dest_obj_2 = "single_conn_dest_2";

  std::ofstream(source_obj_1) << expected_data_1;
  std::ofstream(source_obj_2) << expected_data_2;

  // When
  TransferResult actual_result_1 =
      service2.AsyncGet(source_obj_1, peer1_addr, dest_obj_1).get();
  auto missing_future =
      service2.AsyncGet("non_existent_single_conn_obj", peer1_addr,
                        "non_existent_single_conn_dest");
  EXPECT_THROW(missing_future.get(), std::runtime_error);
  TransferResult actual_result_2 =
      service2.AsyncGet(source_obj_2, peer1_addr, dest_obj_2).get();

  // Then
  EXPECT_TRUE(actual_result_1.success);
  EXPECT_TRUE(actual_result_2.success);
  VerifyFileContentAndRemove(dest_obj_1, expected_data_1);
  VerifyFileContentAndRemove(dest_obj_2, expected_data_2);

  std::remove(source_obj_1.c_str());
  std::remove(source_obj_2.c_str());
  service1.Shutdown();
  service2.Shutdown();
}

// Verifies that simultaneous bidirectional AsyncGet and AsyncPut calls between
// two peers complete without deadlock or stream corruption.
TEST(TransferServiceP2PTest, BidirectionalConcurrentGetAndPut) {
  // Given
  TransferService service1("127.0.0.1");
  int port1 = service1.Initialize();
  ASSERT_GT(port1, 0);

  TransferService service2("127.0.0.1");
  int port2 = service2.Initialize();
  ASSERT_GT(port2, 0);

  std::string addr1 = "127.0.0.1:" + std::to_string(port1);
  std::string addr2 = "127.0.0.1:" + std::to_string(port2);

  const int kNumOps = 8;
  std::vector<std::string> s1_get_sources(kNumOps);
  std::vector<std::string> s2_get_sources(kNumOps);
  std::vector<std::string> s1_get_dests(kNumOps);
  std::vector<std::string> s2_get_dests(kNumOps);
  std::vector<std::string> s1_put_dests(kNumOps);
  std::vector<std::string> s2_put_dests(kNumOps);
  std::vector<std::string> expected_payloads(kNumOps);

  for (int i = 0; i < kNumOps; ++i) {
    expected_payloads[i] =
        "Bidirectional payload " + std::to_string(i) + std::string(1024, 'X');
    s1_get_sources[i] = "bidir_s1_src_" + std::to_string(i);
    s2_get_sources[i] = "bidir_s2_src_" + std::to_string(i);
    s1_get_dests[i] = "bidir_s1_get_dst_" + std::to_string(i);
    s2_get_dests[i] = "bidir_s2_get_dst_" + std::to_string(i);
    s1_put_dests[i] = "bidir_s1_put_dst_" + std::to_string(i);
    s2_put_dests[i] = "bidir_s2_put_dst_" + std::to_string(i);

    std::ofstream(s1_get_sources[i]) << expected_payloads[i];
    std::ofstream(s2_get_sources[i]) << expected_payloads[i];
  }

  // When
  std::vector<std::future<TransferResult>> futures;
  for (int i = 0; i < kNumOps; ++i) {
    futures.push_back(
        service1.AsyncGet(s2_get_sources[i], addr2, s1_get_dests[i]));
    futures.push_back(
        service2.AsyncGet(s1_get_sources[i], addr1, s2_get_dests[i]));
    futures.push_back(
        service1.AsyncPut(const_cast<char*>(expected_payloads[i].data()),
                          expected_payloads[i].size(), addr2, s2_put_dests[i]));
    futures.push_back(
        service2.AsyncPut(const_cast<char*>(expected_payloads[i].data()),
                          expected_payloads[i].size(), addr1, s1_put_dests[i]));
  }

  // Then
  for (auto& fut : futures) {
    TransferResult actual_result = fut.get();
    EXPECT_TRUE(actual_result.success);
  }

  for (int i = 0; i < kNumOps; ++i) {
    VerifyFileContentAndRemove(s1_get_dests[i], expected_payloads[i]);
    VerifyFileContentAndRemove(s2_get_dests[i], expected_payloads[i]);
    VerifyFileContentAndRemove(s1_put_dests[i], expected_payloads[i]);
    VerifyFileContentAndRemove(s2_put_dests[i], expected_payloads[i]);
    std::remove(s1_get_sources[i].c_str());
    std::remove(s2_get_sources[i].c_str());
  }

  service1.Shutdown();
  service2.Shutdown();
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
