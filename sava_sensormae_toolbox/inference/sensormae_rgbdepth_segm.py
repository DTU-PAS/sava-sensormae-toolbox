"""RGB + Depth semantic-segmentation model (placeholder).

This module is a stub for future implementation.  Once an ONNX model for
RGB + Depth segmentation is available, fill in ``_preprocessing``,
``_inference``, and ``_postprocessing`` following the same pattern used in
:class:`SensorMAESegm_RGBThermal`.
"""

import logging

import numpy as np

from .segm_base import SensorMAESegmentation
from ..structures import DetectionListResult

logger = logging.getLogger(__name__)


class SensorMAESegm_RGBDepth(SensorMAESegmentation):
    """SensorMAE segmentation with RGB + depth inputs (not yet implemented)."""

    MODALITY_X_NAME = "depth"

    def __init__(self, runtime, **kwargs):
        super().__init__(runtime, **kwargs)

    def _preprocessing(self, rgb_image: np.ndarray, modality_x_image: np.ndarray):
        raise NotImplementedError("RGB + Depth segmentation is not implemented yet.")

    def _inference(self, preprocessed):
        raise NotImplementedError("RGB + Depth segmentation is not implemented yet.")

    def _postprocessing(self, outputs):
        raise NotImplementedError("RGB + Depth segmentation is not implemented yet.")

