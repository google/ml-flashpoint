#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

# Activate virtual environment
source .venv/bin/activate

# Clean up previous coverage runs
rm -rf python-coverage.xml cxx-coverage.xml htmlcov build

# Re-install with C++ coverage enabled
pip install -e '.[dev]' --config-settings=cmake.args="-DENABLE_COVERAGE=ON"

# Run Python tests with coverage
coverage run --source=src/ml_flashpoint --branch -m pytest -v -s
coverage xml -o python-coverage.xml

coverage report



# To run directly in the terminal, can run with these options to just show missing test coverage:
#  pytest --cov=src/ml_flashpoint --cov-branch --cov-report=term-missing term:skip-covered

# Create C++ coverage report directory
mkdir -p htmlcov/cpp

# Run gcovr for C++ coverage
gcovr --root=. --filter=src/ml_flashpoint --exclude=".*/_deps/.*" --gcov-executable=gcov --txt-metric branch --html-details htmlcov/cpp/index.html --xml-pretty -o cxx-coverage.xml --sort uncovered-number --gcov-ignore-parse-errors=negative_hits.warn,suspicious_hits.warn

# Generate Python HTML report
coverage html -d htmlcov/python

# Generate combined coverage report (TODO: not working)
#mkdir -p htmlcov/combined
#gcovr --add-tracefile python-coverage.xml --add-tracefile cxx-coverage.xml --html-details -o htmlcov/combined/index.html --xml-pretty -o combined-coverage.xml

echo "Coverage reports generated successfully."
echo "Python HTML report: htmlcov/python/index.html"
echo "C++ HTML report: htmlcov/cpp/index.html"
#echo "Combined HTML report: htmlcov/combined/index.html"
