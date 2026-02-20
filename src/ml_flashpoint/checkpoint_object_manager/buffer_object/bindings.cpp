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
#include <pybind11/stl.h>
#include <sys/mman.h>

#include <optional>
#include <string>

#include "buffer_object.h"

namespace py = pybind11;

// Module entry point
PYBIND11_MODULE(buffer_object_ext, m) {
  m.doc() = "C++ Buffer Object";

  py::class_<BufferObject> buffer_object_class(m, "BufferObject",
                                               py::buffer_protocol());

  buffer_object_class
      // We release the GIL for constructors because they perform file I/O and
      // memory mapping, which can be slow. This allows other Python threads
      // to run concurrently. The C++ constructor does not interact with Python
      // objects, making it safe to release the GIL.
      .def(py::init<const std::string&, size_t, bool>(), py::arg("object_id"),
           py::arg("capacity"), py::arg("overwrite") = false,
           py::call_guard<py::gil_scoped_release>())

      .def(py::init<const std::string&>(), py::arg("object_id"),
           py::call_guard<py::gil_scoped_release>())

      .def("close", &BufferObject::close, "Closes the buffer object.",
           py::arg("truncate_size") = std::nullopt,
           py::call_guard<py::gil_scoped_release>())

      .def("get_id", &BufferObject::get_id, "Gets the buffer object id.",
           py::call_guard<py::gil_scoped_release>())

      .def("get_capacity", &BufferObject::get_capacity,
           "Gets the buffer object capacity.",
           py::call_guard<py::gil_scoped_release>())

      .def("resize", &BufferObject::resize, "Resizes the buffer object.",
           py::arg("new_capacity"), py::call_guard<py::gil_scoped_release>())

      .def(
          "get_data_ptr",
          [](const BufferObject& self) -> std::uintptr_t {
            // Safety check: prevent access if the buffer is already closed
            if (self.is_closed()) {
              throw std::runtime_error(
                  "Buffer is closed: cannot access data pointer.");
            }
            // Cast the void* to a standard generic integer type capable of
            // holding a pointer
            return reinterpret_cast<std::uintptr_t>(self.get_data_ptr());
          },
          "Returns the raw memory address of the buffer as an integer.",
          py::call_guard<py::gil_scoped_release>())

      .def_property_readonly("closed", &BufferObject::is_closed,
                             py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("is_readonly", &BufferObject::is_readonly,
                             py::call_guard<py::gil_scoped_release>())

      // The GIL is NOT released for the buffer protocol implementation.
      // This lambda function creates a py::buffer_info object, which is a
      // Python object wrapper. Interacting with Python's C API, including the
      // creation of such objects, requires the GIL to be held to ensure thread
      // safety.
      .def_buffer([](BufferObject& b) -> py::buffer_info {
        if (b.is_closed()) {
          throw std::runtime_error(
              "Cannot create buffer view on a closed BufferObject");
        }

        return py::buffer_info(
            b.get_data_ptr(),                          // Pointer to buffer
            sizeof(uint8_t),                           // Size of one item
            py::format_descriptor<uint8_t>::format(),  // Python struct-style
                                                       //   format descriptor
            1,                                         // Number of dimensions
            {b.get_capacity()},                        // Buffer dimensions
            {sizeof(uint8_t)},                         // Strides (for 1D array)
            b.is_readonly()                            // Readonly flag
        );
      });
}