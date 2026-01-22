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

#include "transfer_helpers.h"

#include <array>
#include <iomanip>
#include <random>
#include <sstream>

namespace ml_flashpoint::replication::transfer_service {

// TODO: Use /dev/urandom or boost library instead of manaully generate uuid.
std::string GenerateUuid() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  std::array<unsigned char, 16> data;
  for (auto& byte : data) {
    byte = dis(gen);
  }

  data[6] = (data[6] & 0x0F) | 0x40;
  data[8] = (data[8] & 0x3F) | 0x80;

  std::stringstream ss;
  ss << std::hex << std::setfill('0');

  int i = 0;
  for (const auto& byte : data) {
    ss << std::setw(2) << static_cast<int>(byte);
    if (i == 3 || i == 5 || i == 7 || i == 9) {
      ss << "-";
    }
    i++;
  }
  return ss.str();
}

}  // namespace ml_flashpoint::replication::transfer_service