_base_ = [
    "../_base_/models/resnet50.py",
    "../_base_/datasets/imagenet_bs32.py",
    "../_base_/schedules/imagenet_bs256.py",
    "../_base_/default_runtime.py",
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", scale=224),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=256, edge="short"),
    dict(type="CenterCrop", crop_size=224),
    dict(type="PackInputs"),
]

data_root = "./data/flower_dataset"
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type="CustomDataset",
        data_root=data_root,
        ann_file="",
        data_prefix="train",
        pipeline=train_pipeline,
        _delete_=True,
    ),
)
val_dataloader = dict(
    dataset=dict(
        type="CustomDataset",
        data_root=data_root,
        ann_file="",
        data_prefix="val",
        pipeline=test_pipeline,
        _delete_=True,
    )
)
test_dataloader = dict(
    dataset=dict(
        type="CustomDataset",
        data_root=data_root,
        ann_file="",
        data_prefix="test",
        pipeline=test_pipeline,
        _delete_=True,
    )
)

train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=10)
