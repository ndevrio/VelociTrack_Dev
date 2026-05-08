import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sys
import os
import time
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from mmapdataset import RealSenseDataset
from models_new import Depth1DCNN, TimeSeriesData

from lightning import seed_everything

import logging
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

np.set_printoptions(precision=4)


def main():
    ########################################
    ###  Process and featurize the data  ###
    ########################################
    #n_splits = 10
    #skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    #lst_accu_stratified = np.zeros((n_splits,))
    #cmat = np.zeros((n_splits, len(class_names), len(class_names)))
    #s = 0

    seed_everything(42)

    torch.set_printoptions(sci_mode=False)

    train_dataset = RealSenseDataset(sys.argv[1], sys.argv[2], sys.argv[3], 0)
    test_dataset = RealSenseDataset(sys.argv[1], sys.argv[2], sys.argv[3], 1)

    model = Depth1DCNN()
    data = TimeSeriesData(train_dataset, test_dataset)
    log_time = time.strftime("%Y%m%d-%H%M%S")
    log_name = "realsense_" + sys.argv[2] + "_" + sys.argv[3] # camera
    logger = TensorBoardLogger('tb_logs', name=log_name)

    with open('output_train_ht.txt', 'a') as f:
        f.write(str(sys.argv[1]) + ' ' + str(sys.argv[2]) + ' ' + str(sys.argv[3]) + '\n')
        
        ########################
        ###  Model training  ###
        ########################
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # Metric to monitor
            dirpath='checkpoints/',  # Directory to save checkpoints
            filename=log_time + "_" + sys.argv[2] + "_" + sys.argv[3] + '_{epoch:02d}-{val_loss:.2f}',  # Checkpoint file name format
            save_top_k=1,  # Save only the best model
            mode='min',  # Mode for monitoring the metric (min for loss, max for accuracy, etc.)
            save_on_train_epoch_end=True
        )

        trainer = pl.Trainer(accelerator="gpu", max_epochs=25, logger=logger, deterministic=True, enable_checkpointing=True, check_val_every_n_epoch=2, callbacks=[checkpoint_callback])
        trainer.fit(model, data)

        ##########################
        ###  Model evaluation  ###
        ##########################
        trainer.test(ckpt_path="best", dataloaders=DataLoader(test_dataset, batch_size=128, shuffle=False)) #model, data)#

        out_cmat = model.cmat.compute().to('cpu').numpy()
        print(out_cmat)
        grasp_acc = out_cmat[1, 1] / (out_cmat[1, 0] + out_cmat[1, 1])

        f.write(str(out_cmat) + "\n")
        f.write(str(grasp_acc) + "\n")
        f.write(trainer.checkpoint_callback.best_model_path + "\n")


    """print("============== Final average ==============")
    print(lst_accu_stratified)
    print("============== Confusion matrix ==============")
    print(np.mean(cmat, axis=0)*10)

    print("============== Saving model ==============")
    torch.save(model.state_dict(), 'models/best_4' + target_dir)
    print("Saved.")"""


if __name__ == "__main__":
    main()
