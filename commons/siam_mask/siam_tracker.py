from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from commons.datatypes import Detection
from commons.file_management import get_local_path_of_url
from commons.siam_mask.experiments.siammask_sharp.custom import Custom
from commons.siam_mask.test import siamese_init, siamese_track
from commons.siam_mask.utils.config_helper import load_config
from commons.siam_mask.utils.load_helper import load_pretrain


class SiamTracker:
    CONFIG_VOT = 'vot'
    CONFIG_VOT_LARGE_DATA = 'vot18'
    CONFIG_DAVIS = 'davis'

    def __init__(self, config=None, device: int = 0):
        if config is None:
            config = self.CONFIG_DAVIS
        self.config = config
        self.model, self.cfg = self.load_model(config)
        self.seq_len = 1
        self.model.eval()

        self.device = None
        self.device_id = None

        self._half = False
        self.set_device(device)

    @staticmethod
    def load_model(config):
        config_path = _CONFIGS[config]
        cfg = load_config(config_path)
        siammask = Custom(anchors=cfg['anchors'])

        resume = get_local_path_of_url(_WEIGHTS[config])
        siammask = load_pretrain(siammask, resume)
        return siammask, cfg

    # noinspection PyMethodOverriding
    def predict(self, imgs: Tuple[np.ndarray, np.ndarray], bboxes: List[Detection],
                calc_mask: bool = True) -> List[Detection]:
        predictions = []

        if len(bboxes) == 0:
            return predictions

        im, next_im = imgs
        boxes = np.array([np.array(det.bbox) for det in bboxes], dtype=np.float32)
        boxes[:, [2, 3]] -= boxes[:, [0, 1]]
        boxes[:, [0, 1]] += boxes[:, [2, 3]] / 2
        boxes = boxes[(boxes[:, 2] >= 1) & (boxes[:, 3] >= 1)]  # ignore boxes which are less than 1x1 pixels

        if len(boxes) == 0:
            return predictions

        states = [{} for _ in range(len(boxes))]
        for i in range(len(boxes)):
            with torch.no_grad():
                states[i] = siamese_init(im, boxes[i][[0, 1]], boxes[i][[2, 3]], self.model, self.cfg['hp'],
                                         self.device)
                states[i] = siamese_track(states[i], next_im, calc_mask, True, self.device)
        if calc_mask:
            masks = np.zeros([len(states), *im.shape[:2]])
            for i, state in enumerate(states):
                masks[i] = state['mask']
            rescaled_mask = ((masks - np.mean(masks, axis=(1, 2), keepdims=True))
                             / np.std(masks, axis=(1, 2), keepdims=True))
            top_preds = rescaled_mask == np.max(rescaled_mask, axis=0, keepdims=True)
            masks *= (np.greater(masks, states[0]['p'].seg_thr) & top_preds).astype(np.float32)
            for det, mask in zip(bboxes, masks):
                _mask = mask > 0
                points = cv2.findNonZero(_mask.astype(np.uint8))
                x, y, w, h = cv2.boundingRect(points)
                predictions.append(Detection(det.name, [x, y, x + w, y + h]))
        else:
            for det, state in zip(bboxes, states):
                x, y = state['target_pos']
                w, h = state['target_sz']
                x -= w // 2
                y -= h // 2
                predictions.append(Detection(det.name, [x, y, x + w, y + h]))

        return predictions

    def set_device(self, device_id: Optional[int]) -> None:
        """
        Moves model and all future inputs to the `device_id` device

        Args:
            device_id: number of the gpu to be used for computations, using cpu if None

        """
        if torch.cuda.is_available() and device_id is not None:
            self.device_id = device_id
            new_device = torch.device('cuda:{}'.format(self.device_id))
        else:
            new_device = torch.device('cpu')

        if new_device != self.device:
            self.model.to(new_device)
        self.device = new_device

    def half(self) -> None:
        """
        Forces the model to perform all gpu computations in float16 dtype
        """
        self._half = True

    def float(self) -> None:
        """
        Forces the model to perform all gpu computations in float32 dtype
        """
        self._half = False


_configs_path = Path('commons/siam_mask/experiments/siammask_sharp')

_CONFIGS = {
    SiamTracker.CONFIG_DAVIS: _configs_path / 'config_davis.json',
    SiamTracker.CONFIG_VOT: _configs_path / 'config_vot.json',
    SiamTracker.CONFIG_VOT_LARGE_DATA: _configs_path / 'config_vot18.json'
}

_WEIGHTS = {
    SiamTracker.CONFIG_DAVIS: 'http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth',
    SiamTracker.CONFIG_VOT: 'http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth',
    SiamTracker.CONFIG_VOT_LARGE_DATA: 'http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth'
}

assert len(_CONFIGS.keys() & _WEIGHTS) == len(_CONFIGS)
