# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import json
from os.path import exists


def proccess_loss(cfg):
    if 'reg' not in cfg:
        cfg['reg'] = {'loss': 'L1Loss'}
    else:
        if 'loss' not in cfg['reg']:
            cfg['reg']['loss'] = 'L1Loss'

    if 'cls' not in cfg:
        cfg['cls'] = {'split': True}

    cfg['weight'] = cfg.get('weight', [1, 1, 36])  # cls, reg, mask


def add_default(conf, default):
    default.update(conf)
    return default


def load_config(config_path):
    assert exists(config_path), '"{}" not exists'.format(config_path)
    config = json.load(open(config_path))

    # deal with network
    if 'network' not in config:
        print('Warning: network lost in config. This will be error in next version')

        config['network'] = {}

        raise Exception('no arch provided')

    # deal with loss
    if 'loss' not in config:
        config['loss'] = {}

    proccess_loss(config['loss'])

    # deal with lr
    if 'lr' not in config:
        config['lr'] = {}
    default = {
        'feature_lr_mult': 1.0,
        'rpn_lr_mult': 1.0,
        'mask_lr_mult': 1.0,
        'type': 'log',
        'start_lr': 0.03
    }
    default.update(config['lr'])
    config['lr'] = default

    # clip
    if 'clip' in config:
        if 'clip' not in config:
            config['clip'] = {}
        if config['clip']['feature'] != config['clip']['rpn']:
            config['clip']['split'] = True

    return config
