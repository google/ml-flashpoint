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

#include <fcntl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "gtest/gtest.h"
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

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
