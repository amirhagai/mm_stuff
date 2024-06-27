_base_ = [
    './__base_fast__/default_runtime.py', './__base_fast__/schedule_3x.py',
    './__base_fast__/dota_rr_con_test_on_data.py'
]
checkpoint =  'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa
# checkpoint = '/root/.cache/torch/hub/checkpoints/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'

import sys 
sys.path.append('/mm_stuff')

custom_imports = dict(imports=['contrastive_support', 'transform.transforms'], allow_failed_imports=False)
# from transform.transforms import *
# from backbones_grad.cspNextGrad import *

# custom_imports = dict(imports=['converters.converter1'], allow_failed_imports=False)
# conv = dict(type='Converter1', a=5, b=6)
# custom_imports = dict(imports=['backbones_grad'], allow_failed_imports=False)
# custom_imports = dict(imports=['transform.transforms', 'backbones_grad.cspNextGrad'], allow_failed_imports=False)
# custom_imports = dict(imports=['backbones_grad.cspNextGrad'], allow_failed_imports=False)
# custom_imports = dict(imports=['transform.transforms', 'backbones_grad.cspNextGrad', 'transform', 'backbones_grad'], allow_failed_imports=False)

# backbone2=dict(
#     type='mmdet.CSPNeXtGrad',
#     arch='P5',
#     expand_ratio=0.5,
#     deepen_factor=1,
#     widen_factor=1,
#     channel_attention=True,
#     norm_cfg=dict(type='SyncBN'),
#     act_cfg=dict(type='SiLU'),

#     )

num_classes_removed = 0 # len(_base_['ignore_classes'].split(' ')) if _base_['ignore_classes'] != '' else 0
trained_model_full_path = "/work_dirs/FT_rotated_rtmdet_l-3x-dota/epoch_2.pth"

random_alpha_y_channel=False
angle_version = 'le90'
model = dict(
    type='RTMDetContrastive',
    data_preprocessor=dict(
        type='DetDataPreprocessorContrastive',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        batch_augments=None),
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)
        ),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHeadContrastive',
        num_classes=15 - num_classes_removed,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=1),
        max_per_img=2000),
)

# batch_size = (2 GPUs) x (4 samples per GPU) = 8
train_dataloader = dict(batch_size=1, num_workers=1)
experiment_name = ''