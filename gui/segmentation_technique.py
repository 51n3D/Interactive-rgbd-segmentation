from abc import ABC, abstractmethod
import numpy as np


class SegmentationTechnique(ABC):
    @abstractmethod
    def segment(self, pos_interactions: tuple, neg_interactions: tuple) -> np.array:
        pass
