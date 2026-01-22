## On startup in this directory

1. If a virtual environment does not exist at `.venv` or is not already activated, create one in `.venv`, prompting the user for confirmation of the command first.
1. Activate the venv via `source .venv/bin/activate`.

## General Rules
1. Always, always, always reread the code in case other external changes have been made to it.
Do not assume that it is in the exact same state as it was the last time you read or edited it.
1. When searching for application code, ignore files and paths listed in .gitignore, especially `.venv`.
Only look at `.venv` when trying to understand dependencies.
1. Use existing code as an example for new code where possible for consistency, but not as a hard rule.
1. Always ensure formatting rules are adhered to by checking pyproject.toml for the configuration, such as line length.
1. Run `ruff check .` to validate formatting after any code changes have been made, and fix any errors using `ruff check --fix .`, and if that's not enough, ask the user if they want to run `ruff format .`.
1. Run `find src -name '*.cpp' -o -name '*.h' | xargs clang-format -i` to format all C++ files. And run `find src -name '*.cpp' -o -name '*.h' | xargs clang-format --dry-run --Werror` to validate the format of C++ files.
1. Run python tools directly, as opposed to via `python3 -m <tool>` e.g. run `pytest ...` instead of `python3 -m pytest ...`.

## Documentation Rules
1. When writing docstrings, always format them in Google style.
1. Write docstrings that explain what the class or function does, how it is expected to be used, and any particular limitations or assumptions it makes.
1. Avoid obvious comments in docstrings like just saying "Initializes the <class name>." for an __init__ function.
If the function's logic is obvious, do not write a redundant one-liner description for the method or function, just document the other aspects of the function, like the parameters and expectations for them.

## Testing Rules
1. Format tests such that the setup, action and assertion parts are separated by an empty line when possible.
Add the comments "# Given", "# When" and "# Then" before each section respectively.
If the action and assertion are combined, can use the comment "# When/Then".
1. So long as the project is using pytest, never import `unittest` or any sub-package of it, and only use pytest mocking capabilities.
1. When naming the expected and actual data variables in a test, use those prefixes: `expected_` and `actual_` respectively, to clearly denote what those variables represent.
1. Minimize mocking.
If something can be done without mocking, that keeps the unit test simple (such as creating inputs that are the actual types instead of mocks), then do that.
This is usually the case for data classes or simple classes that are easy to instantiate.
1. DO NOT remove unrelated test code or assertions when adding or updating a test case.
1. Add a short docstring to each test function explaining what it is testing, i.e. what behavior the test is validating.
