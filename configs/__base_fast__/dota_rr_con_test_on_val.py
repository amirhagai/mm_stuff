
# import sys 
# sys.path.append('/mm_stuff')
# custom_imports = dict(imports=['converters.converter1'], allow_failed_imports=False)
# conv = dict(type='Converter1', a=5, b=6)
custom_imports = dict(imports=['transform.transforms'], allow_failed_imports=False)


# dataset settings
dataset_type = 'DOTADataset'
data_root = '/data/split_ss_dota/fast_test'

backend_args = None


size = 512
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='BboxColorJitterContrastive', prob=1.,class_num=4),
    dict(type='MMDetResizeContrastive', scale=(size, size), keep_ratio=True),
    dict(
        type='RandomFlipContrastive',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(
        type='RandomRotateContrastive',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=[9, 11]),
    dict(
        type='PadContrastive', size=(size, size),
        pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputsContrastive')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(size, size), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(size, size),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(size, size), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(size, size),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=val_pipeline))


test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=val_pipeline))

val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.1, 0.5, 0.8])
test_evaluator = val_evaluator

# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
# test_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     persistent_workers=False,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(img_path='test/images/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='DOTAMetric',
#     format_only=True,
#     merge_patches=True,
#     outfile_prefix='./work_dirs/rtmdet_r/Task1')
