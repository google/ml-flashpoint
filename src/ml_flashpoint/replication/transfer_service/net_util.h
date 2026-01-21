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

#ifndef ML_FLASHPOINT_REPLICATION_NET_UTIL_H_
#define ML_FLASHPOINT_REPLICATION_NET_UTIL_H_

#include <string>

#include "absl/status/statusor.h"
#include "protocol.h"

namespace ml_flashpoint::replication::transfer_service {

// Gets the local primary IPv4 address by iterating network interfaces.
// Returns the IP address as a string, or an error status on failure.
absl::StatusOr<std::string> GetLocalIpAddress();

// Sends exactly data_size bytes from data_ptr over the socket.
// Handles short counts and signals. Returns absl::OkStatus() on success.
//
// Args:
//   sockfd: The socket file descriptor.
//   data_ptr: A pointer to the data to send.
//   data_size: The number of bytes to send.
//
// Returns:
//   absl::OkStatus() on success, or an error status on failure.
absl::Status SendAll(int sockfd, const void* data_ptr, ssize_t data_size);

// Receives exactly data_size bytes into data_ptr from the socket.
// Handles short counts and signals. Returns absl::OkStatus() on success.
//
// Args:
//   sockfd: The socket file descriptor.
//   data_ptr: A pointer to the buffer to receive data into.
//   data_size: The number of bytes to receive.
//
// Returns:
//   absl::OkStatus() on success, or an error status on failure.
absl::Status RecvAll(int sockfd, void* data_ptr, ssize_t data_size);

// Receives an ObjInfoHeader from the socket and ensures its string fields are
// null-terminated.
absl::Status RecvHeader(int sockfd, ObjInfoHeader& header);

}  // namespace ml_flashpoint::replication::transfer_service

#endif  // ML_FLASHPOINT_REPLICATION_NET_UTIL_H_
