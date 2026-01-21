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

#include <future>

#include "object_manager.h"

namespace py = pybind11;

PYBIND11_MODULE(object_manager_ext, m) {
  m.doc() = "Pybind11 bindings for the object_manager C++ library";

  // Defining a basic Future-like class, called BasicFutureVoid, that exposes a
  // wait() method to optionally block on the underlying std::future via its
  // wait() method. The wait method releases the GIL so that other threads can
  // execute while waiting.
  py::class_<std::future<void>>(m, "BasicFutureVoid")
      .def("wait", &std::future<void>::wait, "Wait for the future to complete.",
           py::call_guard<py::gil_scoped_release>());

  // Static function that deletes directories async, return a BasicFutureVoid
  // that can be waited-on.
  m.def("delete_directories_async",
        &ml_flashpoint::checkpoint_object_manager::object_manager::
            delete_directories_async,
        "Asynchronously deletes a list of directories.");
}
