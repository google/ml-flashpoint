# Scripts

This directory contains various adhoc scripts, some for one-off tasks like analyzing logs, some for build tasks.

For clear segregation from our actual library requirements, these scripts have their own `requirements.txt`, and it is highly recommended to use a dedicated venv for them to avoid polluting the project's venv:

```bash
# From the project root
python3 -m venv ./scripts/.venv

source ./scripts/.venv/bin/activate

./scripts/<somescript.py>
```

All dependencies for all scripts should be added to `requirements.txt` - pinned versions are recommended if they matter, but since these are adhoc scripts, not strictly required.

Scripts can specify a shebang so they can be run directly, e.g. (as the very first line): `#!/usr/bin/env python3`.

In any case, the script should provide usage help, or at least have comments specifying how to run them.
