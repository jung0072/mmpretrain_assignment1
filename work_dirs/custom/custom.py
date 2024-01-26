auto_scale_lr = dict(base_batch_size=1024)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
data_root = '/Users/minkijung/Documents/3AISD/Vision/assignment1/data/flower_dataset'
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='small_075',
        norm_cfg=dict(eps=1e-05, momentum=0.1, type='BN'),
        type='MobileNetV3'),
    head=dict(
        act_cfg=dict(type='HSwish'),
        dropout_rate=0.2,
        in_channels=432,
        init_cfg=dict(
            bias=0.0, layer='Linear', mean=0.0, std=0.01, type='Normal'),
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        mid_channels=[
            1024,
        ],
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='StackedLinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    optimizer=dict(
        alpha=0.9,
        eps=0.0316,
        lr=0.064,
        momentum=0.9,
        type='RMSprop',
        weight_decay=1e-05))
param_scheduler = dict(by_epoch=True, gamma=0.973, step_size=2, type='StepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='',
        data_prefix='test',
        data_root=
        '/Users/minkijung/Documents/3AISD/Vision/assignment1/data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=10)
train_dataloader = dict(
    batch_size=32,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='',
        data_prefix='train',
        data_root=
        '/Users/minkijung/Documents/3AISD/Vision/assignment1/data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(pad_val=[
                    104,
                    116,
                    124,
                ]),
                policies='imagenet',
                type='AutoAugment'),
            dict(
                erase_prob=0.2,
                fill_color=[
                    103.53,
                    116.28,
                    123.675,
                ],
                fill_std=[
                    57.375,
                    57.12,
                    58.395,
                ],
                max_area_ratio=0.3333333333333333,
                min_area_ratio=0.02,
                mode='rand',
                type='RandomErasing'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(pad_val=[
            104,
            116,
            124,
        ]),
        policies='imagenet',
        type='AutoAugment'),
    dict(
        erase_prob=0.2,
        fill_color=[
            103.53,
            116.28,
            123.675,
        ],
        fill_std=[
            57.375,
            57.12,
            58.395,
        ],
        max_area_ratio=0.3333333333333333,
        min_area_ratio=0.02,
        mode='rand',
        type='RandomErasing'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=128,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        ann_file='',
        data_prefix='val',
        data_root=
        '/Users/minkijung/Documents/3AISD/Vision/assignment1/data/flower_dataset',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        type='CustomDataset'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/custom'
