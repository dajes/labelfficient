from typing import List, Tuple

from PIL import Image


class BasicPredictor:
    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def track(self, prev_img: Image, cur_img: Image, rel_boxes: List[Tuple[float, float, float, float]],
              prev_classes: List[str]) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
        """
        prev_img: previous image
        cur_img: current image for which we want to predict boxes
        rel_boxes: boxes in range [0;1] with (x1, y1, x2, y2) coordinates in the previous image
        prev_classes: classes for each box in the previous image

        returns: boxes in range [0;1] with (x1, y1, x2, y2) coordinates in the current image and their classes
        """
        return rel_boxes, prev_classes
