from typing import List
import numpy as np  

class DectObject:
    """
    A class to represent a detected object.

    Attributes:
    _xywh : List[float] - 2D bounding box
    _xyzwhd : List[float] - 3D bounding box
    _segm : List[float] - Segmentation mask
    _class_id : str - Class ID of the detected object
    _score : float - Confidence score of the detection
    _pc : List[float] - Point cloud data
    """

    def __init__(
        self,
        xywh: List[float] = [],  # 2D bounding box
        xyzwhd: List[float] = [],  # 3D bounding box
        segm: List[float] = [],  # Segmentation mask
        score: float = 0.0,  # Confidence score
        class_id: str = "",  # Class ID
        pc: List[float] = [],  # Point cloud data
        full_image_segm: np.ndarray = np.array([]),  # Full image segmentation mask
    ) -> None:
        self._xywh: List[float] = xywh
        self._xyzwhd: List[float] = xyzwhd
        self._segm: List[float] = segm
        self._full_image_segm: np.ndarray = np.array([])  # Full image segmentation mask
        self._class_id: str = class_id
        self._score: float = score
        self._pc: List[float] = pc
        self._full_image_segm = full_image_segm

    @property
    def xywh(self) -> List[float]:
        """Returns the 2D bounding box coordinates."""
        return self._xywh

    @xywh.setter
    def xywh(self, value: List[float]) -> None:
        """Sets the 2D bounding box coordinates."""
        self._xywh = value

    @property
    def xyzwhd(self) -> List[float]:
        """Returns the 3D bounding box coordinates."""
        return self._xyzwhd

    @xyzwhd.setter
    def xyzwhd(self, value: List[float]) -> None:
        """Sets the 3D bounding box coordinates."""
        self._xyzwhd = value

    @property
    def segm(self) -> List[float]:
        """Returns the segmentation mask."""
        return self._segm

    @segm.setter
    def segm(self, value: List[float]) -> None:
        """Sets the segmentation mask."""
        self._segm = value

    @property
    def class_id(self) -> str:
        """Returns the class ID of the detected object."""
        return self._class_id

    @class_id.setter
    def class_id(self, value: str) -> None:
        """Sets the class ID of the detected object."""
        self._class_id = value

    @property
    def score(self) -> float:
        """Returns the confidence score of the detection."""
        return self._score

    @score.setter
    def score(self, value: float) -> None:
        """Sets the confidence score of the detection."""
        self._score = value

    @property
    def pc(self) -> List[float]:
        """Returns the point cloud data."""
        return self._pc

    @pc.setter
    def pc(self, value: List[float]) -> None:
        """Sets the point cloud data."""
        self._pc = value

    @property
    def full_image_segm(self) -> np.ndarray:
        """Returns the full image segmentation mask."""
        return self._full_image_segm
    
    @full_image_segm.setter
    def full_image_segm(self, value: np.ndarray) -> None:
        """Sets the full image segmentation mask."""
        self._full_image_segm = value


class DetectionListResult:
    """
    A class to represent a list of detected objects.

    Attributes:
    _dectobjs : List[DectObject] - List of detected objects
    """

    def __init__(self) -> None:
        self._dectobjs: List[DectObject] = []

    def __getitem__(self, idx: int) -> "DectObject":
        return self._dectobjs[idx]

    def __len__(self) -> int:
        return len(self._dectobjs)

    def __iter__(self):
        return iter(self._dectobjs)
    
    @property
    def dectobjs(self) -> List[DectObject]:
        """Returns the list of detected objects."""
        return self._dectobjs

    @dectobjs.setter
    def dectobjs(self, value: List[DectObject]) -> None:
        """Sets the list of detected objects."""
        self._dectobjs = value

    def append(self, dectobj: DectObject) -> None:
        """Appends a detected object to the list."""
        self._dectobjs.append(dectobj)

    def getbboxes(self) -> List[List[float]]:
        """Returns a list of 2D bounding box coordinates for all detected objects."""
        return [dectobj.xywh for dectobj in self._dectobjs]

    def getclassids(self) -> List[str]:
        """Returns a list of class IDs for all detected objects."""
        return [dectobj.class_id for dectobj in self._dectobjs]

    def getscores(self) -> List[float]:
        """Returns a list of confidence scores for all detected objects."""
        return [dectobj.score for dectobj in self._dectobjs]

    def getsegms(self) -> List[List[float]]:
        """Returns a list of segmentation masks for all detected objects."""
        return [dectobj.segm for dectobj in self._dectobjs]

    def getcocoformat(self) -> List[dict]:
        """Returns a list of COCO format annotations for all detected objects."""
        return [
            {
                "bbox": dectobj.xywh,
                "category_id": dectobj.class_id,
                "score": dectobj.score,
                "segmentation": dectobj.segm,
                "area": 0.0,
                "iscrowd": 0,
            }
            for dectobj in self._dectobjs
        ]
