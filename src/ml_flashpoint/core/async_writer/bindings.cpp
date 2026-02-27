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

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <future>

#include "checkpoint_writer.h"
#include "pybind11_futures.h"

namespace py = pybind11;

PYBIND11_MODULE(async_writer_ext, m) {
  m.doc() = "C++ Async Checkpoint Writer Extension";

  using namespace ml_flashpoint::core::async_writer;

  // Result structure for a single write operation.
  py::class_<CppWriteResult>(m, "CppWriteResult")
      .def_readonly("index_id", &CppWriteResult::index_id)
      .def_readonly("size_in_bytes", &CppWriteResult::size_in_bytes)
      .def_readonly("relative_path", &CppWriteResult::relative_path)
      .def_readonly("offset", &CppWriteResult::offset)
      .def_readonly("length", &CppWriteResult::length);

  // Input structure for a tensor write item.
  py::class_<CppTensorWriteItem>(m, "CppTensorWriteItem")
      .def(py::init<int, std::string, at::Tensor>(), py::arg("index_id"),
           py::arg("header_bytes"), py::arg("tensor"));

  // Input structure for a bytes write item.
  py::class_<CppBytesWriteItem>(m, "CppBytesWriteItem")
      .def(py::init<int, std::string>(), py::arg("index_id"), py::arg("data"));

  // Input structure for an object write bucket.
  py::class_<CppObjectWriteBucket>(m, "CppObjectWriteBucket")
      .def(py::init<std::string, std::string, std::vector<CppTensorWriteItem>,
                    std::vector<CppBytesWriteItem>>(),
           py::arg("object_id"), py::arg("object_name"),
           py::arg("tensor_items"), py::arg("bytes_items"));

  // Main entry point for async checkpoint writing from Python.
  // Releases the GIL to allow true parallelism in C++.
  m.def("write_buckets_async", &write_buckets_async, py::arg("buckets"),
        py::arg("initial_buffer_size"), py::arg("format_signature"),
        "Asynchronously writes buckets to buffers. Returns a Python "
        "concurrent.futures.Future.",
        py::call_guard<py::gil_scoped_release>());
}
