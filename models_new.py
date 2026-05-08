import numpy as np
import torch
import math
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, Subset, DataLoader, random_split
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import matplotlib.pyplot as plt


class_names = ['no_touch', 'touch_down']#, 'touch_up']


##########################################
###  Model and DataModule definitions  ###
##########################################
class ResidualMLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = out + x
        return out
    

class Depth1DCNN(pl.LightningModule):
    def __init__(self, ds=1):
        super(Depth1DCNN, self).__init__()
        k_size = 7
        self.cnn1_a = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn2_a = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn3_a = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=k_size, padding='same')
        self.cnn4_a = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k_size, padding='same')
        self.cnn1_b = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn2_b = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn3_b = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=k_size, padding='same')
        self.cnn4_b = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k_size, padding='same')
        self.cnn1_c = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn2_c = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn3_c = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=k_size, padding='same')
        self.cnn4_c = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k_size, padding='same')
        self.cnn1_d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn2_d = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn3_d = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=k_size, padding='same')
        self.cnn4_d = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k_size, padding='same')
        self.cnn1_e = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn2_e = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn3_e = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=k_size, padding='same')
        self.cnn4_e = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k_size, padding='same')

        self.pool = nn.MaxPool1d(2, stride=2)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.res1 = ResidualMLPBlock(512, 512)
        self.res2 = ResidualMLPBlock(512, 512)
        self.res3 = ResidualMLPBlock(512, 512)

        if(ds == 32):
            self.fc1 = nn.Linear(160, 512)
        else:
            self.fc1 = nn.Linear(int(6400/ds), 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

        self.lr=3e-4
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(class_names))
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(class_names))
        self.cmat = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=len(class_names))


    def forward(self, x_in):
        #print(x.size())
        #batch_size, _, _ = x.size()
        #x = x.view(batch_size, -1)
        depth, vel, depth_conv, acc_conv, acc_conv2 = x_in

        print(depth.size())

        # print(np.isnan(depth.cpu()), np.isnan(vel.cpu()), np.isnan(depth_conv.cpu()), np.isnan(acc_conv.cpu()), np.isnan(acc_conv2.cpu()))

        o1 = self.cnn1_a(np.swapaxes(depth, 1, 2))
        o1 = F.relu(o1)
        o1 = self.cnn2_a(o1)
        o1 = self.pool(F.relu(o1))
        o1 = self.cnn3_a(o1)
        o1 = F.relu(o1)
        o1 = self.cnn4_a(o1)
        o1 = self.pool(F.relu(o1))
        o1 = self.drop1(o1)
        
        o2 = self.cnn1_b(np.swapaxes(vel, 1, 2))
        o2 = F.relu(o2)
        o2 = self.cnn2_b(o2)
        o2 = self.pool(F.relu(o2))
        o2 = self.cnn3_b(o2)
        o2 = F.relu(o2)
        o2 = self.cnn4_b(o2)
        o2 = self.pool(F.relu(o2))
        o2 = self.drop1(o2)
        
        o3 = self.cnn1_c(np.swapaxes(depth_conv, 1, 2))
        o3 = F.relu(o3)
        o3 = self.cnn2_c(o3)
        o3 = self.pool(F.relu(o3))
        o3 = self.cnn3_c(o3)
        o3 = F.relu(o3)
        o3 = self.cnn4_c(o3)
        o3 = self.pool(F.relu(o3))
        o3 = self.drop1(o3)
        
        o4 = self.cnn1_d(np.swapaxes(acc_conv, 1, 2))
        o4 = F.relu(o4)
        o4 = self.cnn2_d(o4)
        o4 = self.pool(F.relu(o4))
        o4 = self.cnn3_d(o4)
        o4 = F.relu(o4)
        o4 = self.cnn4_d(o4)
        o4 = self.pool(F.relu(o4))
        o4 = self.drop1(o4)

        o5 = self.cnn1_e(np.swapaxes(acc_conv2, 1, 2))
        o5 = F.relu(o5)
        o5 = self.cnn2_e(o5)
        o5 = self.pool(F.relu(o5))
        o5 = self.cnn3_e(o5)
        o5 = F.relu(o5)
        o5 = self.cnn4_e(o5)
        o5 = self.pool(F.relu(o5))
        o5 = self.drop1(o5)

        o1 = torch.flatten(o1, 1, 2)
        o2 = torch.flatten(o2, 1, 2)
        o3 = torch.flatten(o3, 1, 2)
        o4 = torch.flatten(o4, 1, 2)
        o5 = torch.flatten(o5, 1, 2)

        x = torch.cat((o1, o2, o3, o4, o5), 1)

        x = self.fc1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        preds = self.forward(x)[:, 0]

        loss = self.loss(preds, y)

        binary_pred = torch.sigmoid(preds) >= 0.5
        self.train_accuracy(binary_pred, y)

        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": self.train_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        preds = self.forward(x)[:, 0]
        loss = self.loss(preds, y)

        binary_pred = torch.sigmoid(preds) >= 0.5
        self.cmat.update(binary_pred, y)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        preds = self.forward(x)[:, 0]

        loss = self.loss(preds, y)
        binary_pred = torch.sigmoid(preds) >= 0.5
        accuracy = (binary_pred == y).float().mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss
    

class Acc1DCNN(pl.LightningModule):
    def __init__(self):
        super(Acc1DCNN, self).__init__()
        k_size = 7
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=k_size, padding='same')
        self.cnn3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=k_size, padding='same')
        self.cnn4 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=k_size, padding='same')

        self.pool = nn.MaxPool1d(2, stride=2)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

        self.res1 = ResidualMLPBlock(512, 512)
        self.res2 = ResidualMLPBlock(512, 512)
        self.res3 = ResidualMLPBlock(512, 512)

        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

        self.lr=3e-4
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(class_names))
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(class_names))
        self.cmat = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=len(class_names))

    def forward(self, x_in):
        o1 = self.cnn1(np.swapaxes(x_in, 1, 2))
        o1 = F.relu(o1)
        o1 = self.cnn2(o1)
        o1 = self.pool(F.relu(o1))
        o1 = self.cnn3(o1)
        o1 = F.relu(o1)
        o1 = self.cnn4(o1)
        o1 = self.pool(F.relu(o1))
        o1 = self.drop1(o1)

        x = torch.flatten(o1, 1, 2)

        x = self.fc1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        preds = self.forward(x)[:, 0]

        loss = self.loss(preds, y)

        binary_pred = torch.sigmoid(preds) >= 0.5
        self.train_accuracy(binary_pred, y)

        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": self.train_accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        preds = self.forward(x)[:, 0]
        loss = self.loss(preds, y)

        binary_pred = torch.sigmoid(preds) >= 0.5
        self.cmat.update(binary_pred, y)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        preds = self.forward(x)[:, 0]

        loss = self.loss(preds, y)
        binary_pred = torch.sigmoid(preds) >= 0.5
        accuracy = (binary_pred == y).float().mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss


class TimeSeriesData(pl.LightningDataModule):
    def __init__(self, train_data, test_data):
        super(TimeSeriesData, self).__init__()

        val_split = 0.1
        l1, l2 = math.floor(len(train_data)*val_split), math.ceil(len(train_data)*(1-val_split))
        self.val_data, self.train_data = torch.utils.data.random_split(train_data, [l1, l2])

        self.test_data = test_data

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=512, shuffle=True, num_workers=7, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=512, shuffle=False, num_workers=7, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=512, shuffle=False, num_workers=7, pin_memory=True, persistent_workers=True)