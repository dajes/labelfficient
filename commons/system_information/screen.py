from typing import Union, Tuple

from screeninfo import get_monitors

RESOLUTION = [1 << 32, 1 << 32]
for monitor in get_monitors():
    RESOLUTION[0] = min(RESOLUTION[0], monitor.width)
    RESOLUTION[1] = min(RESOLUTION[1], monitor.height)


def get_target_size(image_size, target: Union[int, float, Tuple[int, int]] = 0.9, return_k=False):
    if isinstance(target, float):
        target = (RESOLUTION[0] * target, RESOLUTION[1] * target)
    elif isinstance(target, int):
        target = (target, target)
    k = min(target[0] / image_size[0], target[1] / image_size[1])
    target_size = int(round(image_size[0] * k)), int(round(image_size[1] * k))
    if return_k:
        return target_size, k
    return target_size
