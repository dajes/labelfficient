from itertools import count

import cv2
import numpy as np
import os

from tqdm import tqdm


def video2frames(path, img_format='jpg'):
    cap = cv2.VideoCapture(path)
    frame_path = '.'.join(path.split('.')[:-1])
    os.makedirs(frame_path, exist_ok=True)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digit_count = int(np.ceil(np.log10(frame_count)))
    name_format = '{:0' + str(digit_count) + 'd}.' + img_format
    print(name_format)
    for i in tqdm(count(), total=frame_count):
        read, frame = cap.read()
        if not read:
            break
        cv2.imwrite(os.path.join(frame_path, name_format.format(i)), frame)


if __name__ == '__main__':
    video2frames(r'D:\videos\videoplayback.webm')
