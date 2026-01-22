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

#ifndef ML_FLASHPOINT_REPLICATION_PROTOCOL_H_
#define ML_FLASHPOINT_REPLICATION_PROTOCOL_H_

#include <sys/socket.h>
#include <sys/types.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

namespace ml_flashpoint::replication::transfer_service {

enum class MessageType : uint8_t {
  kPutObj = 1,
  kGetObj = 2,
  kRespondToGetObj =
      3,  // It's still a Put task, but it's repsond to a get request.
  kAck = 4,
  kError = 5,
};

// string null terminator character.
constexpr char kNullTerm = '\0';

struct ObjInfoHeader {
  MessageType type;
  char source_obj_id[1024];
  char dest_obj_id[1024];
  char source_address[64];
  char dest_address[64];
  char task_id[64];
  ssize_t obj_size;

  ObjInfoHeader() : type(MessageType::kAck), obj_size(0) {
    source_obj_id[0] = kNullTerm;
    dest_obj_id[0] = kNullTerm;
    source_address[0] = kNullTerm;
    dest_address[0] = kNullTerm;
    task_id[0] = kNullTerm;
  }

  // Ensure all char arrays are null-terminated to prevent buffer over-read
  // segfaults when they are converted to std::string or used in logging.
  void NullTerminateCharArrays() {
    source_obj_id[sizeof(source_obj_id) - 1] = kNullTerm;
    dest_obj_id[sizeof(dest_obj_id) - 1] = kNullTerm;
    source_address[sizeof(source_address) - 1] = kNullTerm;
    dest_address[sizeof(dest_address) - 1] = kNullTerm;
    task_id[sizeof(task_id) - 1] = kNullTerm;
  }
} __attribute__((packed));

constexpr size_t kHeaderSize = sizeof(ObjInfoHeader);

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_PROTOCOL_H_
