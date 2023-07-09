import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.dice_score import dice_loss
from utils.dataset import BuildingDataset

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

import segmentation_models_pytorch as smp
from sklearn.model_selection import StratifiedKFold

import sys
from config import *
from loguru import logger
logger.remove()
logger.add(f'{prefix_save}_train.log', format="{time} | {message}")

df = pd.read_csv('train.csv')

skf = StratifiedKFold(n_splits=cv)
for i, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(df)), df['ratio'] > 0)):
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    
    train_dataset = BuildingDataset(train_df, train_transform)
    val_dataset = BuildingDataset(val_df, val_transform)

    train_loader = DataLoader(
        train_dataset, 
        sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        shuffle=False
    )

    # init model
    model = getattr(smp, model_cfg.arch)(**model_cfg.hparams)

    # init optimizer
    optimizer = getattr(optim, opt_cfg.name)(model.parameters(), **opt_cfg.hparams)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                    patience=scheduler_patience, 
                                                    verbose=True) 
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.BCEWithLogitsLoss()

    # train
    model.train()
    max_val_loss = np.infty
    for epoch in range(1, max_epochs + 1):
        total_loss = 0
        n_total = 0
        with tqdm(total=len(train_loader), ncols=150, ascii=True, desc=f'CV {i + 1} Epoch {epoch}') as t:
            # train one epoch
            for images, true_masks in train_loader:
                images = images.to(device)
                true_masks = true_masks.to(device)
                model = model.to(device)

                pred_masks = model(images).squeeze(1)
                loss = criterion(pred_masks, true_masks)
                loss += dice_loss(F.sigmoid(pred_masks), true_masks, multiclass=False)
                
                total_loss += loss.item() * images.shape[0]
                n_total += images.shape[0]

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                avg_loss = total_loss / n_total
                t.update()
                t.set_postfix_str(f'avg loss: {avg_loss: .5f}')
        # val one epoch
        total_loss = 0
        n_total = 0
        with torch.no_grad():
            model.eval()
            for images, true_masks in val_loader:
                images = images.to(device)
                true_masks = true_masks.to(device)
                model = model.to(device)

                pred_masks = model(images).squeeze(1)
                loss = criterion(pred_masks, true_masks)
                loss += dice_loss(F.sigmoid(pred_masks), true_masks, multiclass=False)
                
                total_loss += loss.item() * images.shape[0]
                n_total += images.shape[0]
            avg_val_loss = total_loss / n_total
            scheduler.step(avg_val_loss)
            logger.info(f'CV {i + 1} Epoch {epoch} val_loss: {total_loss / n_total}')
            if avg_val_loss < max_val_loss:
                max_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{prefix_save}_best_cv_{i + 1}.pt')
