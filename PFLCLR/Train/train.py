import os
import copy
import torch
import wandb
import numpy as np
from torch import utils
from pathlib import Path
from Model import XrayCLR
from datetime import datetime
from config import ex, config
import pytorch_lightning as pl
from tokenizers import Tokenizer
from Datasets import IuxrayMultiImageDataset, MimiccxrMultiImageDataset,MimiccxrCombineIuMultiImageDataset
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    if _config['wandb_log']:
        with open(_config['wandb_api_key_fp'], 'r') as f:
            wandb_api_key = f.read().strip()
            wandb.login(key=wandb_api_key)

    datestime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_version = _config['version'] if _config['version'] is not None else datestime_str
    _config['model_dir'] = os.path.join(_config["model_dir"], _config["name"], model_version)
    Path(_config['model_dir']).mkdir(parents=True, exist_ok=True)
    pl.seed_everything(_config["seed"])

    if _config['dataset_name'] == "combine":
        dataset_train = MimiccxrCombineIuMultiImageDataset(_config, split="train")
        dataset_valid = MimiccxrCombineIuMultiImageDataset(_config, split="val")
    elif _config['dataset_name'] == "iuxray":
        dataset_train = IuxrayMultiImageDataset(_config, split="train")
        dataset_valid = IuxrayMultiImageDataset(_config, split="val")
    elif _config['dataset_name'] == "mimic_cxr":
        dataset_train = MimiccxrMultiImageDataset(_config, split="train")
        dataset_valid = MimiccxrMultiImageDataset(_config, split="val")
    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']

    model = XrayCLR(_config)
    model.set_loss_func(_config['loss'])

    train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], shuffle=True, num_workers=_config["data_workers"])
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"])

    checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
        dirpath=_config['model_dir'], 
        filename='loss_{epoch}-{valid_MulticlassAveragePrecision:.3f}-{valid_MulticlassAccuracy:.3f}',
        verbose=True,
        save_top_k=1, 
        every_n_epochs=1,
        monitor="valid_loss", 
        mode="min", 
        save_last=True)

    checkpoint_callback_map = pl.callbacks.ModelCheckpoint(
        dirpath=_config['model_dir'], 
        filename='map_{epoch}-{valid_MulticlassAveragePrecision:.3f}-{valid_MulticlassAccuracy:.3f}',
        verbose=True,
        save_top_k=1, 
        every_n_epochs=1,
        monitor="valid_MulticlassAveragePrecision", 
        mode="max")            
    summary_callback = pl.callbacks.ModelSummary(max_depth=1)
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    csv_logger = pl.loggers.CSVLogger(save_dir=_config["log_dir"], name=_config['name'], version=datestime_str)
    csv_logger.log_hyperparams(_config)
    loggers=[csv_logger]
    if _config['wandb_log']:
        wandb_logger = pl.loggers.WandbLogger(project='XrayCLR', save_dir=_config["log_dir"], name=_config['name'], version=model_version)
        wandb_logger.experiment.config.update(_config, allow_val_change=True)
        loggers.append(wandb_logger)
    trainer = pl.Trainer(max_epochs=_config['max_epochs'], 
                        logger=loggers, 
                        # log_every_n_steps=50, # 
                        # gradient_clip_val=0.5,
                        # limit_train_batches=_config['limit_train_batches'],
                        # limit_val_batches=_config['limit_valid_batches'],
                        log_every_n_steps=(int(len(dataset_train)) // _config['batch_size']) // 3,
                        callbacks=[checkpoint_callback_loss, checkpoint_callback_map, lr_callback, summary_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=_config['ckpt_path'])


    