# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py',
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth'
load_from= '../dino_4scale_internimage_t_1x_coco.pth'
model = dict(
    type='DINO',
    backbone=dict(
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=True,
        out_indices=(1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 256, 512],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DINOHead',
        num_query=5,
        num_classes=5,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='DinoTransformer',
            two_stage_num_proposals=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=5))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', direction=['horizontal'], flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(384, 384)],
                      multiscale_mode='value')
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 384),
        flip=False,
        transforms=[
            dict(type='Resize'),
            dict(type='RandomFlip', direction=['horizontal']),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
# dataset settings
dataset_type = 'CocoDataset'
classes = ('l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1') 
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file="process_data/train_2.json",
        img_prefix='/root/images/Sagittal T1',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file="process_data/val_2.json",
        img_prefix='/root/images/Sagittal T1',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file="process_data/val_2.json",
        img_prefix='/root/images/Sagittal T1',
        pipeline=test_pipeline)
)
runner = dict(type='EpochBasedRunner', max_epochs=16)
# optimizer
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=0.9,
                       depths=[4, 4, 18, 4]))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[14, 15])
evaluation = dict(save_best='auto', interval=2)
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=4,
    save_last=True,
)
resume_from = None

work_dir = 'work_dirs_2'