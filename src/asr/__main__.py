"""``python -m asr`` entry point.

The pipeline module is imported lazily here: importing it from
``asr/__init__.py`` (which it is, for the public API) and then running it with
``-m`` makes runpy warn about a module that is already in ``sys.modules``.
A dedicated ``__main__`` avoids that without hiding the API.
"""

from __future__ import annotations

import sys

from asr.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
