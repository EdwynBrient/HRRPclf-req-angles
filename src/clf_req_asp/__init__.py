"""
Conditional aspect-angle estimation via Kalman filtering for HRRP classification.
Main package initialization.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Conditional HRRP classification with Kalman-based aspect-angle estimation"

from . import models
from . import utils
from . import dataset

__all__ = ["models", "utils", "dataset"]
