from typing import Iterable, Union

import numpy as np


class Detection:
    def __init__(self, name: str, bbox: Iterable[Union[float, int]] = None) -> None:
        self.name = name

        self.bbox = np.round(bbox).astype(int)
