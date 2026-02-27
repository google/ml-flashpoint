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

#include "checkpoint_writer.h"

#include <algorithm>
#include <cstring>
#include <future>
#include <thread>

#include <ATen/Parallel.h>
#include "absl/log/log.h"
#include "buffer_object.h"

namespace ml_flashpoint::core::async_writer {

namespace {

constexpr size_t METADATA_SIZE = 4096;

/**
 * @brief Performs a multi-threaded memory copy if the data size is large enough.
 */
void parallel_memcpy(void* dest, const void* src, size_t n) {
  // Use a 1MB grain size to avoid threading overhead for small tensors.
  const size_t grain_size = 1024 * 1024;
  if (n < grain_size) {
    std::memcpy(dest, src, n);
    return;
  }

  uint8_t* d = static_cast<uint8_t*>(dest);
  const uint8_t* s = static_cast<const uint8_t*>(src);

  // at::parallel_for uses the thread pool configured via at::set_num_threads().
  at::parallel_for(0, n, grain_size, [&](int64_t start, int64_t end) {
    std::memcpy(d + start, s + start, end - start);
  });
}

/**
 * @brief C++ version of the BufferMetadata structure.
 * Must match the Python BufferMetadataType in buffer_metadata.py.
 */
struct BufferMetadata {
  uint64_t len_written_data;
  char format_signature[8];
  uint8_t reserved[METADATA_SIZE - 16];
};

/**
 * @brief The actual synchronous task that performs the writes.
 */
std::vector<CppWriteResult> write_buckets_task(
    const std::vector<CppObjectWriteBucket>& buckets,
    size_t initial_buffer_size, const std::string& format_signature) {
  std::vector<CppWriteResult> all_results;

  for (const auto& bucket : buckets) {
    try {
      // Create or overwrite the buffer object.
      // We add METADATA_SIZE because initial_buffer_size is for the data area.
      BufferObject buffer(bucket.object_id, initial_buffer_size + METADATA_SIZE,
                          true);
      uint8_t* base_ptr = static_cast<uint8_t*>(buffer.get_data_ptr());

      // Initialize metadata block at the beginning of the buffer.
      BufferMetadata* meta = reinterpret_cast<BufferMetadata*>(base_ptr);
      std::memset(meta, 0, METADATA_SIZE);
      std::memcpy(meta->format_signature, format_signature.data(),
                  std::min(format_signature.size(), (size_t)8));

      size_t current_pos = 0;  // Relative to data section (after metadata)

      // Write tensor items: [HeaderLen (4B)] [Header (JSON/Pickle)] [Raw Data]
      for (const auto& item : bucket.tensor_items) {
        size_t item_start_offset = current_pos;

        // Copy serialized TensorHeader
        std::memcpy(base_ptr + METADATA_SIZE + current_pos,
                    item.header_bytes.data(), item.header_bytes.size());
        current_pos += item.header_bytes.size();

        // Copy raw tensor data using parallel copy
        size_t data_size = item.tensor.nbytes();
        if (data_size > 0) {
          parallel_memcpy(base_ptr + METADATA_SIZE + current_pos,
                          item.tensor.data_ptr(), data_size);
          current_pos += data_size;
        }

        size_t item_length = current_pos - item_start_offset;
        all_results.push_back({item.index_id, item_length, bucket.object_name,
                               item_start_offset, item_length});
      }

      // Write raw bytes items (for non-tensor data resolved via SavePlanner)
      for (const auto& item : bucket.bytes_items) {
        size_t item_start_offset = current_pos;

        std::memcpy(base_ptr + METADATA_SIZE + current_pos, item.data.data(),
                    item.data.size());
        current_pos += item.data.size();

        size_t item_length = current_pos - item_start_offset;
        all_results.push_back({item.index_id, item_length, bucket.object_name,
                               item_start_offset, item_length});
      }

      // Update metadata with the total amount of data written.
      meta->len_written_data = current_pos;

      // Close and truncate the file to its actual content size.
      buffer.close(METADATA_SIZE + current_pos);

    } catch (const std::exception& e) {
      LOG(ERROR) << "Error writing bucket " << bucket.object_id << ": "
                 << e.what();
      // In a more robust implementation, we might want to propagate this failure
      // back to Python via the future's exception.
    }
  }

  return all_results;
}

}  // namespace

std::future<std::vector<CppWriteResult>> write_buckets_async(
    std::vector<CppObjectWriteBucket> buckets, size_t initial_buffer_size,
    std::string format_signature) {
  // Use a promise to bridge the async task result back to the std::future.
  auto promise = std::make_shared<std::promise<std::vector<CppWriteResult>>>();
  auto future = promise->get_future();

  // Launch the write task in a detached thread.
  // We move the buckets into the thread to avoid copies.
  std::thread t([promise, buckets = std::move(buckets), initial_buffer_size,
                 format_signature = std::move(format_signature)]() {
    try {
      auto results = write_buckets_task(buckets, initial_buffer_size,
                                        format_signature);
      promise->set_value(std::move(results));
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  });

  t.detach();
  return future;
}

}  // namespace ml_flashpoint::core::async_writer
