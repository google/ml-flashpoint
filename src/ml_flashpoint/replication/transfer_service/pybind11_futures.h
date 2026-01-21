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

#ifndef ML_FLASHPOINT_REPLICATION_PYBIND11_FUTURES_H_
#define ML_FLASHPOINT_REPLICATION_PYBIND11_FUTURES_H_

#include <pybind11/pybind11.h>

#include <exception>
#include <future>
#include <thread>
#include <type_traits>

namespace pybind11 {
namespace detail {

/**
 * @brief A pybind11 type caster to convert std::future<T> into a Python
 * concurrent.futures.Future.
 */
template <typename T>
struct type_caster<std::future<T>> {
 public:
  PYBIND11_TYPE_CASTER(std::future<T>, _("concurrent.futures.Future"));

  static handle cast(std::future<T>&& src, return_value_policy, handle) {
    pybind11::gil_scoped_acquire acquire;
    pybind11::object future_module =
        pybind11::module_::import("concurrent.futures");
    pybind11::object py_future = future_module.attr("Future")();

    std::thread([py_future, cpp_future = std::move(src)]() mutable {
      try {
        T result = cpp_future.get();
        pybind11::gil_scoped_acquire acquire;
        py_future.attr("set_result")(pybind11::cast(result));
        py_future = pybind11::object();
      } catch (const std::exception& e) {
        pybind11::gil_scoped_acquire acquire;
        pybind11::object exception_type =
            pybind11::module_::import("builtins").attr("RuntimeError");
        py_future.attr("set_exception")(exception_type(e.what()));
        py_future = pybind11::object();
      } catch (...) {
        pybind11::gil_scoped_acquire acquire;
        pybind11::object exception_type =
            pybind11::module_::import("builtins").attr("RuntimeError");
        py_future.attr("set_exception")(
            exception_type("Unknown C++ exception caught"));
        py_future = pybind11::object();
      }
    }).detach();

    return py_future.release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // ML_FLASHPOINT_REPLICATION_PYBIND11_FUTURES_H_
