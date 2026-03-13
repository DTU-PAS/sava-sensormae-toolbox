"""Intermediate base class for SensorMAE semantic-segmentation models.

Collects segmentation-specific utilities so that individual modality
leaf classes only need to implement the pipeline methods.
"""

from typing import List

import numpy as np

from .base import Model


class SensorMAESegmentation(Model):
    """Base for all SensorMAE segmentation models (any modality pair)."""

    def __init__(self, runtime, **kwargs):
        super().__init__()
        self.session = runtime
        # Original image dimensions for un-padding the output mask
        self._orig_hw: tuple | None = None
