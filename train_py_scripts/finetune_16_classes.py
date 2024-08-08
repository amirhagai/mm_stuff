# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import numpy as np
import torch
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS, MODELS
from mmengine.runner import Runner, turn_on_activation_checkpointing
from new_steps_for_16_classes import new_optimizer_step_option_1
from mmengine.model import is_model_wrapper
from mmengine.model.efficient_conv_bn_eval import turn_on_efficient_conv_bn_eval
import random
from mmrotate.utils import register_all_modules
from mmengine.device import set_device
import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    
    parser.add_argument('optimization_option', type=int, help='optimization_option')
    
    # 1 -  only the part i change
    # 2 - all classification head, here we can also use the other config file
    # 3 - all the bbox head (cls, reg, ang)
    # 4 - entire model, here we can also use the other config file
    
    parser.add_argument('--device', help='help to set device other then the default'\
        , default="")
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


def test_outs(runner, copied_model):
    test_loop = runner.build_test_loop(runner._test_loop)
    test_loop.runner.model.eval()
    copied_model.eval()
    outs = []
    with torch.no_grad():
        for idx, data_batch in enumerate(test_loop.dataloader):
            # test_loop.run_iter(idx, data_batch)
            with autocast(enabled=test_loop.fp16):
                
                # d1 = test_loop.runner.model.data_preprocessor(data_batch, False)
                # d2 = copied_model.data_preprocessor(data_batch, False)

                # x = test_loop.runner.model.extract_feat(d1['inputs'])
                # outs = test_loop.runner.model.bbox_head(x)
                
                # x2 = copied_model.extract_feat(d2['inputs'])
                # outs2 = copied_model.bbox_head(x2)
                # print((outs[0][0][:, :15, :, :] - outs2[0][0]).abs().sum())
                # print((outs[0][1][:, :15, :, :] - outs2[0][1]).abs().sum())
                # print((outs[0][2][:, :15, :, :] - outs2[0][2]).abs().sum())
                
                
                # batch_img_metas = [
                #     data_samples.metainfo for data_samples in d1['data_samples']
                # ]

                # predictions = test_loop.runner.model.bbox_head.predict_by_feat(
                #     *outs, batch_img_metas=batch_img_metas, rescale=True)
                
                # predictions2 = copied_model.bbox_head.predict_by_feat(
                #     *outs2, batch_img_metas=batch_img_metas, rescale=True)
                
                            
                outputs1 = test_loop.runner.model.test_step(data_batch)
                outputs2 = copied_model.test_step(data_batch)
                diff = (outputs2[0].pred_instances.labels != outputs1[0].pred_instances.labels).sum()
                if diff > 0:
                    print(diff)
            if idx % 10 == 0 and idx > 0:
                print(idx)
        print("done")
        # outs.append(outputs1)
    return outs


def train_model(runner, optimization_option) -> nn.Module:
    """Launch training.

    Returns:
        nn.Module: The model after training.
    """
    if is_model_wrapper(runner.model):
        ori_model = runner.model.module
    else:
        ori_model = runner.model
    assert hasattr(ori_model, 'train_step'), (
        'If you want to train your model, please make sure your model '
        'has implemented `train_step`.')

    if runner._val_loop is not None:
        assert hasattr(ori_model, 'val_step'), (
            'If you want to validate your model, please make sure your '
            'model has implemented `val_step`.')

    if runner._train_loop is None:
        raise RuntimeError(
            '`self._train_loop` should not be None when calling train '
            'method. Please provide `train_dataloader`, `train_cfg`, '
            '`optimizer` and `param_scheduler` arguments when '
            'initializing runner.')

    runner._train_loop = runner.build_train_loop(
        runner._train_loop)  # type: ignore


    if optimization_option == 1 or optimization_option == 2: # train bbox class part only
        for name, param in runner.model.named_parameters():
            if 'bbox_head' not in name or 'cls' not in name:
                param.requires_grad = False

    elif optimization_option == 3: # train bbox class part only
        for name, param in runner.model.named_parameters():
            if 'bbox_head' not in name:
                param.requires_grad = False
    
    elif optimization_option == 4:
        pass # train entire model
    
    else:
        print('optimization option should be one -f - 1, 2, 3, 4')
        return
                

    # `build_optimizer` should be called before `build_param_scheduler`
    #  because the latter depends on the former
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    # Automatically scaling lr by linear scaling rule
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)

    if runner.param_schedulers is not None:
        runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
            runner.param_schedulers)  # type: ignore

    if runner._val_loop is not None:

        loop = runner._val_loop 
        runner._val_loop = runner.build_val_loop(
            runner._val_loop)  # type: ignore
        if runner.train_val_loop_flag:
            dataset = runner._val_dataloader['dataset']
            runner._val_dataloader['dataset'] = runner._train_dataloader['dataset']
            runner._val_dataloader['dataset']['pipeline'] = dataset['pipeline']
            runner._train_val_loop = runner.build_val_loop(
                loop)  # type: ignore
            runner._val_dataloader['dataset'] = dataset
    # TODO: add a contextmanager to avoid calling `before_run` many times
    runner.call_hook('before_run')

    # initialize the model weights
    # runner._init_model_weights()

    # try to enable activation_checkpointing feature
    modules = runner.cfg.get('activation_checkpointing', None)
    if modules is not None:
        runner.logger.info(f'Enabling the "activation_checkpointing" feature'
                            f' for sub-modules: {modules}')
        turn_on_activation_checkpointing(ori_model, modules)

    # try to enable efficient_conv_bn_eval feature
    modules = runner.cfg.get('efficient_conv_bn_eval', None)
    if modules is not None:
        runner.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                            f' for sub-modules: {modules}')
        turn_on_efficient_conv_bn_eval(ori_model, modules)

    # make sure checkpoint-related hooks are triggered after `before_run`
    # runner.load_or_resume()

    # Initiate inner count of `optim_wrapper`.
    runner.optim_wrapper.initialize_count_status(
        runner.model,
        runner._train_loop.iter,  # type: ignore
        runner._train_loop.max_iters)  # type: ignore

    # Maybe compile the model according to options in self.cfg.compile
    # This must be called **AFTER** model has been wrapped.
    runner._maybe_compile('train_step')
    if optimization_option == 1:
        runner.optim_wrapper.step = new_optimizer_step_option_1.__get__(runner.optim_wrapper, runner.optim_wrapper.__class__) 

    model = runner.train_loop.run()  # type: ignore
    runner.call_hook('after_run')
    return model


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    # cfg.custom_hooks = None
    # cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    runner.train_val_loop_flag = True
    runner.model.bbox_head.cls_out_channels = 16
    runner.model.bbox_head.num_classes = 16
    # runner.register_hooks.ema_model
    
    runner.call_hook('before_run')
    runner.load_checkpoint(args.checkpoint, map_location=args.device)
    copied_model = copy.deepcopy(runner.model)
    # runner._hooks[1].ema_model = copied_model
    
    
    new_rtm_cls = nn.ModuleList()
    for layer in runner.model.bbox_head.rtm_cls:
        # Step 2: Create a new convolutional layer with 16 output channels instead of 15
        new_layer = nn.Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1), device=args.device)
        
        # Initialize the new_layer weights with zeros or another preferred method
        nn.init.normal_(new_layer.weight, mean=layer.weight.mean().item(), std=layer.weight.std().item())
        nn.init.constant_(new_layer.bias, layer.bias.mean().item() * 1.1)
        
        # Step 3: Copy the weights and biases from the old layer to the new layer for the first 15 channels
        with torch.no_grad():
            new_layer.weight[:15, :, :, :] = layer.weight.clone()
            new_layer.bias[:15] = layer.bias.clone()

        # Add the newly created layer to the new ModuleList
        new_rtm_cls.append(new_layer)
        
    # copied_model = copy.deepcopy(runner.model)


    # Step 4: Replace the old rtm_cls ModuleList with the new one
    runner.model.bbox_head.rtm_cls = new_rtm_cls
    runner.model.bbox_head.cls_out_channels = 16
    
    runner._hooks[1].ema_model = MODELS.build(
        runner._hooks[1].ema_cfg, default_args=dict(model=runner.model))
    # test_outs(runner, copied_model)
    # start training
    # runner.val_loop.run()

    # runner.val_loop.run()
    i = 0
    for name, param in runner.model.named_parameters():
        param.param_name = name
        print(name);i+=1
    print(i)
    
    
    # runner.optim_wrapper.step = new_step.__get__(runner.optim_wrapper, runner.optim_wrapper.__class__) 
    
    # model = runner.train()
    train_model(runner, optimization_option=args.optimization_option)
    print()


if __name__ == '__main__':
    main()




