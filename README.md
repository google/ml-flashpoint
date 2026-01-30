# ML Flashpoint

[![Build and Test](https://github.com/google/ml-flashpoint/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/google/ml-flashpoint/actions/workflows/build_and_test.yml)

A memory-first, lightning-fast, ready-to-use ML checkpointing library.

PyTorch DCP, Megatron-LM and NeMo 2.0 adapters are readily available for seamless integration, on top of our general core checkpointing library that is available for custom integrations.

If interested in direct integration support with another framework, let us know! 
Please create a [feature request](https://github.com/google/ml-flashpoint/issues/new?template=feature_request.md), or upvote an existing one.

For learning more about using the library and its performance, check out the [user documentation](https://google.github.io/ml-flashpoint). 
Below you will find development instructions for contributors.

## Installation

This library defines core dependencies, as well as additional optional dependencies for specific adapters, to avoid polluting consumers with unnecessary dependencies.
See the adapters installation commands for examples of the available adapters.

### Core Library
```bash
pip install -e .
```

To avoid building C++ tests (and pulling test dependencies), such as when using for production:

```bash
pip install -e . --config-settings=cmake.define.BUILD_TESTING=OFF
```

NOTE: Currently C++ binaries are expected to be in the package alongside the code, so editable mode (`-e`) is used.

### With Adapters
```bash
# PyTorch
pip install -e .[pytorch]

# Megatron-LM
pip install -e .[megatron]

# Multiple
pip install -e .[pytorch,megatron]
```

## Development

### Python version

Ensure you have the correct Python version.
As of this writing, the project uses Python 3.10, due to limitations in NeMo's dependencies.

To confirm, see which versions of python come up when tab-completing `python` in your shell.

You could install `pyenv` to manage different Python versions: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation.

And then install the desired Python version with it e.g. `pyenv install 3.10`.

### Build and Installation

NOTE: If you already have a `.venv` for a different version in this repository, run `rm -rf .venv` first.

To set up the development environment, run (at the project root):

```bash
# Create and activate a virtual environment (e.g., using venv) in your local env (only needed once, but is safe to rerun)
python3.10 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode with development dependencies
pip install -e .[dev]
```

### Linting and Testing

All code changes **must** be accompanied by comprehensive unit tests, and integration tests where feasible.
With AI coding tools, there's no good reason to cut corners or omit tests.
You can prompt your coding tool to "create a comprehensive test plan for X, covering edge cases and corner cases" that you can review.

*   **Build C++ Components:** The C++ components are built automatically when you run one of the `pip install` commands from above.
*   **Python Format:** To apply automated fixes, run (with caution):
    NOTE: This may also modify lines that _do not_ violate the lint rules, so use cautiously!
    ```bash
    ruff check --fix .
    ruff format .
    ```

*   **Python Lint:** To check for code style violations, run:
    ```bash
    ruff check .
    ```

*   **C++ Format:** To apply automated fixes, run:

    ```bash
    # install clang-format
    sudo apt-get update && sudo apt-get install -y clang-format

    # format all C++ files
    find src -name '*.cpp' -o -name '*.h' | xargs clang-format -i
    ```

*   **C++ Lint:** Check for style violations, run:
    ```bash
    find src -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run --Werror
    ```

*   **Test:** To run all tests (Python and C++), run:
    ```bash
    pytest
    ```
    * Python tests should be in the `tests` directory, in a package matching the subject-under-test, and the test files should start with `test_`.
    * C++ tests should be in a `test` directory next to the subject-under-test (so within the `src` directory), and should _end_ with `_test.cpp`.

#### Code Coverage

To calculate code coverage, run `./run_coverage.sh` from the project root.
It will activate the venv located at `.venv`, remove build files, re-install the project, and produce coverage reports.

### Conventional Commits

This project uses [conventional commits](https://www.conventionalcommits.org/), and the commit message should complete the sentence: "This change will...".
Specifying the scope for commits is optional, but highly recommended.
Typically, the scope will match the package the change relates to, and can use `/` for sub-packages, e.g.:

```
chore(replication): add the ReplicationManager skeleton class

feat(adapter/nemo): implement the callback to trigger MLFlashpoint checkpoints
```

## Releases

We use release tags of the form `vX.Y.Z` for production releases, following [semver](https://semver.org/), starting with [zerover](https://0ver.org/).

Releases should be created as GitHub Releases, which can be done [here](https://github.com/google/ml-flashpoint/releases/new).

The helper script `create_release.py` will generate release notes that can be added to the Release.

Command: `./scripts/create_release.py`.
Add `-h` for help.

Requirements:

* These release tags **MUST** be immutable - they cannot be modified or deleted after they are created.
* These release tags **MUST** be created from an approved and merged commit, typically from the `main` branch.
* They **MUST NOT** be created from unapproved, unmerged commits, such as a feature branch or patchset.
The commit used to create the release tag must always be accessible and not temporary.

## User Documentation Site

User documentation is all maintained in the `docs/` directory, and is generated using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/getting-started/).
See the `.example-syntax.md` file for guidance on certain supported syntax.

When making changes, you can view them locally via `mkdocs serve`.

Once changes are merged to `main`, they are automatically deployed to the documentation site, available at https://google.github.io/ml-flashpoint.
