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

#ifndef ML_FLASHPOINT_CHECKPOINT_WRITER_H
#define ML_FLASHPOINT_CHECKPOINT_WRITER_H

#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include <torch/extension.h>

namespace ml_flashpoint::core::async_writer {

/**
 * @brief Represents a tensor item to be written to a checkpoint buffer.
 */
struct CppTensorWriteItem {
  int index_id;              // Local mapping ID to the Python MetadataIndex
  std::string header_bytes;  // Serialized TensorHeader
  at::Tensor tensor;         // The tensor payload (ensures memory lifetime)
};

/**
 * @brief Represents a bytes item to be written to a checkpoint buffer.
 */
struct CppBytesWriteItem {
  int index_id;      // Local mapping ID to the Python MetadataIndex
  std::string data;  // Raw bytes data
};

/**
 * @brief Container for writes to a single object ID.
 */
struct CppObjectWriteBucket {
  std::string object_id;    // Full path/ID of the object
  std::string object_name;  // Relative name for storage metadata
  std::vector<CppTensorWriteItem> tensor_items;
  std::vector<CppBytesWriteItem> bytes_items;
};

/**
 * @brief Result of a single item write operation.
 */
struct CppWriteResult {
  int index_id;                // Local mapping ID to the Python MetadataIndex
  size_t size_in_bytes;        // Total bytes written for this item
  std::string relative_path;   // Object name
  size_t offset;               // Offset within the object's data section
  size_t length;               // Length of the item's data
};

/**
 * @brief Asynchronously writes buckets to buffers.
 *
 * This function performs the actual data copying in a separate C++ thread,
 * bypassing the Python GIL.
 */
std::future<std::vector<CppWriteResult>> write_buckets_async(
    std::vector<CppObjectWriteBucket> buckets, size_t initial_buffer_size,
    std::string format_signature);

}  // namespace ml_flashpoint::core::async_writer

#endif  // ML_FLASHPOINT_CHECKPOINT_WRITER_H
