"""Shared fixtures for the speech pipeline test suite."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asr.config import load_config
