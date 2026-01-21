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
#include <ifaddrs.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <memory>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "protocol.h"

namespace ml_flashpoint::replication::transfer_service {

constexpr ssize_t kSendChunkSize = 1024 * 1024;  // 1MB
constexpr ssize_t kRecvChunkSize = 1024 * 1024;  // 1MB

namespace {

// Helper to get the MTU for a given interface name.
// Returns -1 on failure.
int GetInterfaceMtu(int sock, const char* if_name) {
  ifreq ifr = {};
  strncpy(ifr.ifr_name, if_name, IFNAMSIZ - 1);
  if (ioctl(sock, SIOCGIFMTU, &ifr) < 0) {
    return -1;
  }
  return ifr.ifr_mtu;
}

}  // namespace

// TODO: Add more tests && consult gcp team about how to handle this.
absl::StatusOr<std::string> GetLocalIpAddress() {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) {
    return absl::InternalError(
        absl::StrFormat("getifaddrs() failed: %s", strerror(errno)));
  }
  // Use a custom deleter with unique_ptr to ensure freeifaddrs is called.

  std::unique_ptr<struct ifaddrs, void (*)(struct ifaddrs*)> ifaddr_ptr(
      ifaddr, freeifaddrs);

  int sock = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock < 0) {
    return absl::InternalError("Failed to create socket for ioctl");
  }
  auto sock_closer = absl::MakeCleanup([sock] { close(sock); });

  std::string best_ip;
  int max_score = -1;

  // Iterate through the linked list of network interfaces.

  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    // Basic filtering: must be up, have an address, be IPv4, and not loopback.

    if (ifa->ifa_addr == nullptr || !(ifa->ifa_flags & IFF_UP) ||
        ifa->ifa_addr->sa_family != AF_INET ||
        (ifa->ifa_flags & IFF_LOOPBACK)) {
      continue;
    }

    int score = 0;
    // Prefer interfaces that are likely to be physical NICs.

    if (absl::StartsWith(ifa->ifa_name, "eth") ||
        absl::StartsWith(ifa->ifa_name, "eno") ||
        absl::StartsWith(ifa->ifa_name, "ens")) {
      score += 100;
    }
    // Strongly prefer interfaces with higher MTUs.

    int mtu = GetInterfaceMtu(sock, ifa->ifa_name);
    if (mtu > 0) {
      score += mtu;
    }

    if (score > max_score) {
      max_score = score;
      std::array<char, INET_ADDRSTRLEN> ip_buffer;
      void* sin_addr =
          &reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr)->sin_addr;
      const char* result =
          inet_ntop(AF_INET, sin_addr, ip_buffer.data(), ip_buffer.size());

      if (result != nullptr) {
        best_ip = result;
      }
    }
  }

  if (best_ip.empty()) {
    return absl::NotFoundError("No suitable IPv4 network interface found.");
  }
  return best_ip;
}

// TODO: Upgrate to c++ 20 and use std::byte, std::span.
absl::Status SendAll(int sockfd, const void* data_ptr, ssize_t data_size) {
  const char* current_ptr = static_cast<const char*>(data_ptr);
  ssize_t remaining_bytes = data_size;
  while (remaining_bytes > 0) {
    ssize_t bytes_to_send = std::min(remaining_bytes, kSendChunkSize);
    ssize_t sent_this_call =
        send(sockfd, current_ptr, bytes_to_send, MSG_NOSIGNAL);
    if (sent_this_call >= 0) {
      current_ptr += sent_this_call;
      remaining_bytes -= sent_this_call;
    } else {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        struct pollfd pfd = {.fd = sockfd, .events = POLLOUT};
        int ret = poll(&pfd, 1, -1);
        if (ret < 0) {
          return absl::ErrnoToStatus(
              errno, "poll failed while waiting for socket to be writable");
        }
        continue;
      }
      return absl::ErrnoToStatus(errno, "Send failed");
    }
  }
  return absl::OkStatus();
}

absl::Status RecvAll(int sockfd, void* data_ptr, ssize_t data_size) {
  char* current_ptr = static_cast<char*>(data_ptr);
  ssize_t remaining_bytes = data_size;
  while (remaining_bytes > 0) {
    ssize_t bytes_to_recv = std::min(remaining_bytes, kRecvChunkSize);
    ssize_t received = recv(sockfd, current_ptr, bytes_to_recv, 0);
    if (received > 0) {
      current_ptr += received;
      remaining_bytes -= received;
    } else if (received == 0) {
      return absl::UnavailableError(absl::StrFormat(
          "Sender closed connection gracefully. Received %ld bytes out of %ld "
          "expected.",
          data_size - remaining_bytes, data_size));
    } else {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        struct pollfd pfd = {.fd = sockfd, .events = POLLIN};
        int ret = poll(&pfd, 1, -1);
        if (ret < 0) {
          return absl::ErrnoToStatus(
              errno, "poll failed while waiting for socket to be readable");
        }
        continue;
      }
      return absl::ErrnoToStatus(errno, "Recv failed");
    }
  }
  return absl::OkStatus();
}

absl::Status RecvHeader(int sockfd, ObjInfoHeader& header) {
  absl::Status status = RecvAll(sockfd, &header, kHeaderSize);
  if (status.ok()) {
    header.NullTerminateCharArrays();
  }
  return status;
}

}  // namespace ml_flashpoint::replication::transfer_service
