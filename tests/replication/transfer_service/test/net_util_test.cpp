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

#include "net_util.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <thread>  // Required for std::thread
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ml_flashpoint::replication::transfer_service {
namespace {

// Helper to set a socket to non-blocking mode.
void SetNonBlocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  ASSERT_NE(flags, -1);
  ASSERT_NE(fcntl(fd, F_SETFL, flags | O_NONBLOCK), -1);
}

class SendRecvTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds_), 0);
  }

  void TearDown() override {
    close(fds_[0]);
    close(fds_[1]);
  }

  int fds_[2];
};

TEST_F(SendRecvTest, SendAndRecvAll_Basic) {
  // Given
  std::string sent_data = "hello world";
  std::vector<char> recv_buffer(sent_data.size());

  // When
  absl::Status send_status =
      SendAll(fds_[0], sent_data.data(), sent_data.size());
  ASSERT_TRUE(send_status.ok());

  absl::Status recv_status =
      RecvAll(fds_[1], recv_buffer.data(), recv_buffer.size());
  ASSERT_TRUE(recv_status.ok());

  // Then
  std::string received_data(recv_buffer.begin(), recv_buffer.end());
  EXPECT_EQ(sent_data, received_data);
}

TEST_F(SendRecvTest, SendAndRecvAll_NonBlocking) {
  // Given
  SetNonBlocking(fds_[0]);
  SetNonBlocking(fds_[1]);

  std::string sent_data(1024 * 1024, 'a');  // 1MB of data
  std::vector<char> recv_buffer(sent_data.size());

  // Use a separate thread for receiving to avoid deadlock.
  std::thread reader_thread([this, &recv_buffer]() {
    absl::Status recv_status =
        RecvAll(fds_[1], recv_buffer.data(), recv_buffer.size());
    ASSERT_TRUE(recv_status.ok());
  });

  // When
  absl::Status send_status =
      SendAll(fds_[0], sent_data.data(), sent_data.size());
  ASSERT_TRUE(send_status.ok());

  reader_thread.join();

  // Then
  std::string received_data(recv_buffer.begin(), recv_buffer.end());
  EXPECT_EQ(sent_data, received_data);
}

TEST_F(SendRecvTest, RecvAll_IncompleteRead) {
  // Given
  std::string sent_data = "short message";
  std::vector<char> recv_buffer(sent_data.size() + 5);  // Larger buffer

  absl::Status send_status =
      SendAll(fds_[0], sent_data.data(), sent_data.size());
  ASSERT_TRUE(send_status.ok());

  // Close the sending socket to signal EOF after data is sent.
  close(fds_[0]);

  // When
  absl::Status recv_status =
      RecvAll(fds_[1], recv_buffer.data(), recv_buffer.size());

  // Then
  EXPECT_FALSE(recv_status.ok());
  EXPECT_EQ(recv_status.code(), absl::StatusCode::kUnavailable);
}

TEST_F(SendRecvTest, RecvHeader_Basic) {
  // Given
  ObjInfoHeader sent_header;
  sent_header.type = MessageType::kPutObj;
  std::strcpy(sent_header.dest_obj_id, "test_id");
  sent_header.obj_size = 1234;

  ASSERT_EQ(send(fds_[0], &sent_header, sizeof(sent_header), 0),
            sizeof(sent_header));

  ObjInfoHeader recv_header;

  // When
  absl::Status status = RecvHeader(fds_[1], recv_header);

  // Then
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(recv_header.type, sent_header.type);
  EXPECT_STREQ(recv_header.dest_obj_id, sent_header.dest_obj_id);
  EXPECT_EQ(recv_header.obj_size, sent_header.obj_size);
}

TEST_F(SendRecvTest, RecvHeader_FailsOnClosedSocket) {
  // Given
  close(fds_[0]);

  ObjInfoHeader recv_header;

  // When
  absl::Status status = RecvHeader(fds_[1], recv_header);

  // Then
  EXPECT_FALSE(status.ok());
}

TEST_F(SendRecvTest, RecvHeader_NullTerminatesCharArrays) {
  // Given
  ObjInfoHeader sent_header;
  std::memset(sent_header.dest_obj_id, 'A', sizeof(sent_header.dest_obj_id));

  ASSERT_EQ(send(fds_[0], &sent_header, sizeof(sent_header), 0),
            sizeof(sent_header));

  ObjInfoHeader recv_header;

  // When
  absl::Status status = RecvHeader(fds_[1], recv_header);

  // Then
  EXPECT_TRUE(status.ok());
  for (size_t i = 0; i < sizeof(recv_header.dest_obj_id) - 1; ++i) {
    EXPECT_EQ(recv_header.dest_obj_id[i], 'A') << "at index " << i;
  }
  EXPECT_EQ(recv_header.dest_obj_id[sizeof(recv_header.dest_obj_id) - 1], '\0');
}

TEST(NetUtilTest, GetLocalIpAddress) {
  absl::StatusOr<std::string> status_or_ip = GetLocalIpAddress();
  ASSERT_TRUE(status_or_ip.ok()) << status_or_ip.status();
  const std::string& ip_address = status_or_ip.value();
  EXPECT_FALSE(ip_address.empty());

  // Buffers to hold the binary representation of the address.
  struct in_addr ipv4_addr;
  struct in6_addr ipv6_addr;

  // Try to parse the string as a valid IPv4 address.
  // inet_pton returns 1 on success.
  const bool is_valid_ipv4 =
      inet_pton(AF_INET, ip_address.c_str(), &ipv4_addr) == 1;

  // Try to parse the string as a valid IPv6 address.
  const bool is_valid_ipv6 =
      inet_pton(AF_INET6, ip_address.c_str(), &ipv6_addr) == 1;

  // The address must be *either* a valid IPv4 or a valid IPv6 format.
  // We use ASSERT_TRUE again, as the following checks depend on this.
  ASSERT_TRUE(is_valid_ipv4 || is_valid_ipv6)
      << "IP address '" << ip_address
      << "' is not a valid IPv4 or IPv6 format.";

  // A GetLocalIpAddress() function should return a specific interface's
  // address, not the "unspecified" or "any" address.
  if (is_valid_ipv4) {
    // Check against INADDR_ANY (0.0.0.0)
    EXPECT_NE(ipv4_addr.s_addr, INADDR_ANY)
        << "Got 'any' address (0.0.0.0), expected a specific interface IP.";
  }

  if (is_valid_ipv6) {
    // Check against in6addr_any (::)
    EXPECT_NE(memcmp(&ipv6_addr, &in6addr_any, sizeof(in6addr_any)), 0)
        << "Got 'any' address (::), expected a specific interface IP.";
  }

  // Note: We are implicitly allowing loopback addresses ('127.0.0.1' or '::1')
  // as they are valid "local" IP addresses.
}

}  // namespace
}  // namespace ml_flashpoint::replication::transfer_service
