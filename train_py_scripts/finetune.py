# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import torch
from mmrotate.utils import register_all_modules
from mmengine.device import set_device
from mmengine.registry import (MODELS, DefaultScope)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    # parser.add_argument('trained_model_full_path', help='path to .pth file')
    parser.add_argument('--device', help='help to set device other then the default'\
        , default="")
    # parser.add_argument('--layers_to_remove', help='which layers to remove'\
    #     , default="bbox_head")
    # parser.add_argument('--exact_layers_or_start_with', help='remove the exact layers or layers start with name'\
    #     , default="start_with")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def load_pre_train_model(cfg):

    runner = Runner.from_cfg(cfg)
    model = MODELS.build(cfg['model'])
    trained = torch.load(cfg['trained_model_full_path'])
    if cfg['exact_layers_or_start_with'] == "start_with":
        trained_trim = {k:v for k, v in trained['state_dict'].items() if not k.startswith(cfg['layers_to_remove'])}
    else:
        to_remove = set(cfg['layers_to_remove'].split(' '))
        trained_trim = {k:v for k, v in trained['state_dict'].items() if not k in to_remove}
    model.load_state_dict(trained_trim , strict=False)
    return model


def main():

    args = parse_args()
    if args.device != "":
        set_device(args.device)

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume

    model = load_pre_train_model(cfg)
    runner = Runner(
    model=model,
    work_dir=cfg['work_dir'],
    train_dataloader=cfg.get('train_dataloader'),
    val_dataloader=cfg.get('val_dataloader'),
    test_dataloader=cfg.get('test_dataloader'),
    train_cfg=cfg.get('train_cfg'),
    val_cfg=cfg.get('val_cfg'),
    test_cfg=cfg.get('test_cfg'),
    auto_scale_lr=cfg.get('auto_scale_lr'),
    optim_wrapper=cfg.get('optim_wrapper'),
    param_scheduler=cfg.get('param_scheduler'),
    val_evaluator=cfg.get('val_evaluator'),
    test_evaluator=cfg.get('test_evaluator'),
    default_hooks=cfg.get('default_hooks'),
    custom_hooks=cfg.get('custom_hooks'),
    data_preprocessor=cfg.get('data_preprocessor'),
    load_from=cfg.get('load_from'),
    resume=cfg.get('resume', False),
    launcher=cfg.get('launcher', 'none'),
    env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
    log_processor=cfg.get('log_processor'),
    log_level=cfg.get('log_level', 'INFO'),
    visualizer=cfg.get('visualizer'),
    default_scope=cfg.get('default_scope', 'mmengine'),
    randomness=cfg.get('randomness', dict(seed=None)),
    experiment_name=cfg.get('experiment_name'),
    cfg=cfg,
)


    model = runner.train()


if __name__ == '__main__':
    main()
