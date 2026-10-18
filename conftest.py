"""Make the repository root importable so that ``from aml...`` works."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
