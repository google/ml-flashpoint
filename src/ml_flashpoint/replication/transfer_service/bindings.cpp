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

#include "pybind11_futures.h"
#include "transfer_service.h"

namespace py = pybind11;

PYBIND11_MODULE(transfer_service_ext, m) {
  m.doc() = "C++ Transfer Service";

  using namespace ml_flashpoint::replication::transfer_service;

  py::class_<TransferResult>(m, "TransferResult")
      .def(py::init<>())
      .def_readonly("task_id", &TransferResult::task_id)
      .def_readonly("success", &TransferResult::success)
      .def_readonly("error_message", &TransferResult::error_message);

  py::class_<TransferService>(m, "TransferService")
      .def(py::init<>())

      // Synchronous methods remain unchanged (still blocking)
      .def("initialize", &TransferService::Initialize, py::arg("listen_port"),
           py::arg("threads") = 16, py::arg("conn_pool_per_peer") = 16,
           py::arg("global_rank") = -1,
           "Initializes and starts the C++ transfer service.")

      .def("shutdown", &TransferService::Shutdown,
           "Stops the C++ transer service.")

      // Async methods return a Future. The Future is a std::future, which is
      // a C++ object that can be used to asynchronously wait for the result of
      // the transfer.
      .def(
          "async_put",
          [](TransferService& self, std::uintptr_t data_ptr, size_t data_size,
             const std::string& dest_address,
             const std::string& dest_object_id) {
            // This manually casts the integer from Python to a C++ pointer.
            void* ptr = reinterpret_cast<void*>(data_ptr);
            return self.AsyncPut(ptr, data_size, dest_address, dest_object_id);
          },
          py::arg("data_ptr"),   // Pointer to the data in memory. Passed as an
                                 // integer from Python.
          py::arg("data_size"),  // Size of the data in bytes.
          py::arg("dest_address"),  // The destination service address, expected
                                    // in "host:port" format (e.g.,
                                    // "127.0.0.1:12345").
          py::arg("dest_object_id"),  // A unique identifier for the object at
                                      // the destination.
          "Asynchronously puts data to a remote peer. Returns a Future.")

      .def("async_get", &TransferService::AsyncGet,
           py::arg("remote_object_id_to_get_obj_from"),  // The ID of the object
                                                         // to request from the
                                                         // remote peer.
           py::arg("source_address"),  // The source service address, expected
                                       // in "host:port" format (e.g.,
                                       // "127.0.0.1:12345").
           py::arg("local_object_id_to_save_received_obj"),  // The ID to assign
                                                             // to the received
                                                             // object locally.
           "Asynchronously requests an object from the specified peer. Returns "
           "a Future.");
}
