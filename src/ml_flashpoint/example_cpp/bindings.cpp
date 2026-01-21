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

#include <pybind11/pybind11.h>

#include "example_cpp.h"

namespace py = pybind11;

PYBIND11_MODULE(example_cpp_ext, m) {
  m.doc() = "Example C++ Bindings";
  m.def("add3", &example_cpp::add3, "Adds three integers", py::arg("a"),
        py::arg("b"), py::arg("c"));
}
