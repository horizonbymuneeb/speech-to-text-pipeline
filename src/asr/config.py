"""Configuration objects for the speech pipeline.

``config.yaml`` holds the tunables. Everything has a default here so the YAML
only needs the keys you actually want to change, and every section is validated
on load: a typo in ``config.yaml`` fails loudly instead of silently running with
the default.

Frame-based settings are declared in milliseconds and converted to sample /
frame counts through methods that take the sample rate, because the same config
should work at 8 kHz and 16 kHz without editing numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml
