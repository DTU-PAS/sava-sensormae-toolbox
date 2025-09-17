
from abc import ABC, abstractmethod


class Model(ABC):
    """This class represents a generic model with different method that every subclass model
        is likely to use and reimplement

    Args:
        ABC (ABC): Abstract class
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self._inference(*args, **kwargs)

    @abstractmethod
    def _inference(self):
        pass

    @abstractmethod
    def _preprocessing(self):
        pass

    @abstractmethod
    def _postprocessing(self):
        pass
