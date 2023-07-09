import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

train_transform = A.Compose([
    A.Flip(),
    A.RandomRotate90(0.5),
    A.Transpose(0.5),
    A.HorizontalFlip(0.5),
    A.VerticalFlip(0.5),
    # Extend
    # A.GaussNoise(p=0.2),
    # A.Perspective(p=0.5),
    # A.OneOf([
    #     A.Sharpen(p=0.8),
    #     A.Blur(blur_limit=3, p=0.8),
    #     A.MotionBlur(blur_limit=3, p=0.8)
    # ]),
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])

batch_size = 32
num_workers = 8
device = 'cuda'

cv = 5
max_epochs = 150

class model_cfg:
    arch = 'UnetPlusPlus'
    hparams = dict(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        in_channels=1,
        classes=1
    )

class opt_cfg:
    name = 'AdamW'
    hparams = dict(
        lr=1e-3
    )

scheduler_patience = 5

prefix_save = f'{model_cfg.arch}_{model_cfg.hparams["encoder_name"]}_cv{cv}'
