import os
import numpy as np
import torch
from torch.utils import data
from random import shuffle

# save .npz files as memory mapped files instead
# load in parts using the following method, accessing parts of files using memory mapped arrays

class RealSenseDataset(data.Dataset):

    def __init__(self, data_dir, participant, trial_mode, test_mode):

        self.files = []
        with open(data_dir, 'r') as data_file:
            for line in data_file:

                ### Trial mode 0:   Train on everything except all data from `participant`
                ### Trial mode 1+:   Train on everything except session 1+ data from `participant`

                check = False
                if(not test_mode):
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and ("s1" in line or (participant + '/') not in line)) or
                        (trial_mode == '2' and ("s2" in line or (participant + '/') not in line)) or
                        (trial_mode == '3' and ("s3" in line or (participant + '/') not in line))):
                        check = True
                else:
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and "s1" in line and (participant + '/') in line) or
                        (trial_mode == '2' and "s2" in line and (participant + '/') in line) or
                        (trial_mode == '3' and "s3" in line and (participant + '/') in line)):
                        check = True

                if(not test_mode and not check):
                    self.files.append(line)
                elif(test_mode and check):
                    self.files.append(line)
        self.data_tuples = []

        self.cam_shape = (320, 8)
        self.cam_offset = self.cam_shape[0]*self.cam_shape[1]

        self.acc_shape = (320, 1)
        self.acc_offset = self.acc_shape[0]*self.acc_shape[1]

        for n in self.files:
            cam_file = n[:-1] + "_cam.bin"
            #acc_file = n[:-1] + "_acc.bin"
            label_file = n[:-1] + "_labels.bin"

            labels = np.memmap(label_file, dtype='ubyte', mode='r')

            # print(len(labels), np.count_nonzero(labels==1)) # check label split

            idxs = [i for i in range(0, labels.shape[0])]

            if len(idxs) == 0:
                continue

            for j in idxs:
                #data_tuple = (cam_file, acc_file, j, labels[j])
                data_tuple = (cam_file, j, labels[j])
                self.data_tuples.append(data_tuple)

        #shuffle(self.data_tuples)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_tuple = self.data_tuples[idx]

        cam_sample = np.memmap(sample_tuple[0], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.cam_offset, shape=self.cam_shape)
        #acc_sample = np.memmap(sample_tuple[1], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.acc_offset, shape=self.acc_shape)
        cam_sample = torch.as_tensor(cam_sample)
        #acc_sample = torch.as_tensor(acc_sample)
        label = float(sample_tuple[2])
        # label = sample_tuple[2]

        depth = cam_sample[:, 0]
        depth = depth.unsqueeze(1)
        vel = cam_sample[:, 1]
        vel = vel.unsqueeze(1)
        depth_conv = cam_sample[:, 3]
        depth_conv = depth_conv.unsqueeze(1)
        acc_conv = cam_sample[:, 4]
        acc_conv = acc_conv.unsqueeze(1)
        acc_conv2 = cam_sample[:, 5]
        acc_conv2 = acc_conv2.unsqueeze(1)

        # X = torch.stack((depth, vel, depth_conv, acc_conv, acc_conv2), dim=1).flatten()
        # X = torch.stack((depth, vel), dim=1).flatten()
        # return X, label
        return (depth, vel, depth_conv, acc_conv, acc_conv2), label


class AccDataset(data.Dataset):

    def __init__(self, data_dir, participant, trial_mode, test_mode):

        self.files = []
        with open(data_dir, 'r') as data_file:
            for line in data_file:

                ### Trial mode 0:   Train on everything except all data from `participant`
                ### Trial mode 1+:   Train on everything except session 1+ data from `participant`

                check = False
                if(not test_mode):
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and ("s1" in line or (participant + '/') not in line)) or
                        (trial_mode == '2' and ("s2" in line or (participant + '/') not in line)) or
                        (trial_mode == '3' and ("s3" in line or (participant + '/') not in line))):
                        check = True
                else:
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and "s1" in line and (participant + '/') in line) or
                        (trial_mode == '2' and "s2" in line and (participant + '/') in line) or
                        (trial_mode == '3' and "s3" in line and (participant + '/') in line)):
                        check = True

                if(not test_mode and not check):
                    self.files.append(line)
                elif(test_mode and check):
                    self.files.append(line)
        self.data_tuples = []

        self.cam_shape = (320, 8)
        self.cam_offset = self.cam_shape[0]*self.cam_shape[1]

        self.acc_shape = (320, 1)
        self.acc_offset = self.acc_shape[0]*self.acc_shape[1]

        for n in self.files:
            # cam_file = n[:-1] + "_cam.bin"
            acc_file = n[:-1] + "_acc.bin"
            label_file = n[:-1] + "_labels.bin"

            labels = np.memmap(label_file, dtype='ubyte', mode='r')

            # print(len(labels), np.count_nonzero(labels==1)) # check label split

            idxs = [i for i in range(0, labels.shape[0])]

            if len(idxs) == 0:
                continue

            for j in idxs:
                #data_tuple = (cam_file, acc_file, j, labels[j])
                data_tuple = (acc_file, j, labels[j])
                self.data_tuples.append(data_tuple)

        #shuffle(self.data_tuples)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_tuple = self.data_tuples[idx]

        # cam_sample = np.memmap(sample_tuple[0], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.acc_offset, shape=self.acc_shape)
        acc_sample = np.memmap(sample_tuple[0], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.acc_offset, shape=self.acc_shape)
        # cam_sample = torch.as_tensor(cam_sample)
        acc_sample = torch.as_tensor(acc_sample)
        label = float(sample_tuple[2])
        # label = sample_tuple[2]

        # X = torch.stack((depth, vel, depth_conv, acc_conv, acc_conv2), dim=1).flatten()
        # X = torch.stack((depth, vel), dim=1).flatten()
        # return X, label
        return acc_sample, label
    

class AblationDataset(data.Dataset):

    def __init__(self, data_dir, participant, trial_mode, test_mode, ds=1):

        self.files = []
        with open(data_dir, 'r') as data_file:
            for line in data_file:

                ### Trial mode 0:   Train on everything except all data from `participant`
                ### Trial mode 1+:   Train on everything except session 1+ data from `participant`

                check = False
                if(not test_mode):
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and ("s1" in line or (participant + '/') not in line)) or
                        (trial_mode == '2' and ("s2" in line or (participant + '/') not in line)) or
                        (trial_mode == '3' and ("s3" in line or (participant + '/') not in line))):
                        check = True
                else:
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and "s1" in line and (participant + '/') in line) or
                        (trial_mode == '2' and "s2" in line and (participant + '/') in line) or
                        (trial_mode == '3' and "s3" in line and (participant + '/') in line)):
                        check = True

                if(not test_mode and not check):
                    self.files.append(line)
                elif(test_mode and check):
                    self.files.append(line)
        self.data_tuples = []
        self.ds = ds

        self.cam_shape = (int(320/ds), 8)
        self.cam_offset = self.cam_shape[0]*self.cam_shape[1]

        self.acc_shape = (int(320/ds), 1)
        self.acc_offset = self.acc_shape[0]*self.acc_shape[1]

        for n in self.files:
            if(self.ds == 1):
                cam_file = n[:-1] + "_cam.bin"
                #acc_file = n[:-1] + "_acc.bin"
                label_file = n[:-1] + "_labels.bin"
            else:
                cam_file = n[:-1] + "_ds_" + str(ds) + "_cam.bin"
                #acc_file = n[:-1] + "_ds_" + str(ds) + "_acc.bin"
                label_file = n[:-1] + "_ds_" + str(ds) + "_labels.bin"

            labels = np.memmap(label_file, dtype='ubyte', mode='r')

            # print(len(labels), np.count_nonzero(labels==1)) # check label split

            idxs = [i for i in range(0, labels.shape[0])]

            if len(idxs) == 0:
                continue

            for j in idxs:
                #data_tuple = (cam_file, acc_file, j, labels[j])
                data_tuple = (cam_file, j, labels[j])
                self.data_tuples.append(data_tuple)

        #shuffle(self.data_tuples)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_tuple = self.data_tuples[idx]

        cam_sample = np.memmap(sample_tuple[0], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.cam_offset, shape=self.cam_shape)
        #acc_sample = np.memmap(sample_tuple[1], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.acc_offset, shape=self.acc_shape)
        cam_sample = torch.as_tensor(cam_sample)
        #acc_sample = torch.as_tensor(acc_sample)

        label = float(sample_tuple[2])
        # label = sample_tuple[2]

        depth = cam_sample[:, 0]
        depth = depth.unsqueeze(1)
        vel = cam_sample[:, 1]
        vel = vel.unsqueeze(1)
        depth_conv = cam_sample[:, 3]
        depth_conv = depth_conv.unsqueeze(1)
        acc_conv = cam_sample[:, 4]
        acc_conv = acc_conv.unsqueeze(1)
        acc_conv2 = cam_sample[:, 5]
        acc_conv2 = acc_conv2.unsqueeze(1)

        # X = torch.stack((depth, vel, depth_conv, acc_conv, acc_conv2), dim=1).flatten()
        # X = torch.stack((depth, vel), dim=1).flatten()
        # return X, label
        return (depth, vel, depth_conv, acc_conv, acc_conv2), label
    

class ClassicDataset(data.Dataset):

    def __init__(self, data_dir, participant, trial_mode, test_mode, ds=1):

        self.files = []
        with open(data_dir, 'r') as data_file:
            for line in data_file:

                ### Trial mode 0:   Train on everything except all data from `participant`
                ### Trial mode 1+:   Train on everything except session 1+ data from `participant`

                check = False
                if(not test_mode):
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and ("s1" in line or (participant + '/') not in line)) or
                        (trial_mode == '2' and ("s2" in line or (participant + '/') not in line)) or
                        (trial_mode == '3' and ("s3" in line or (participant + '/') not in line))):
                        check = True
                else:
                    if(((trial_mode == '0') and (participant + '/') in line) or
                        (trial_mode == '1' and "s1" in line and (participant + '/') in line) or
                        (trial_mode == '2' and "s2" in line and (participant + '/') in line) or
                        (trial_mode == '3' and "s3" in line and (participant + '/') in line)):
                        check = True

                if(not test_mode and not check):
                    self.files.append(line)
                elif(test_mode and check):
                    self.files.append(line)
        self.data_tuples = []
        self.ds = ds

        self.cam_shape = (int(320/ds), 8)
        self.cam_offset = self.cam_shape[0]*self.cam_shape[1]

        self.features_shape = (1,)
        self.features_offset = self.features_shape[0]

        self.acc_shape = (int(320/ds), 1)
        self.acc_offset = self.acc_shape[0]*self.acc_shape[1]

        for n in self.files:
            if(self.ds == 1):
                feat_file = n[:-1] + "_features.bin"
                #acc_file = n[:-1] + "_acc.bin"
                label_file = n[:-1] + "_labels.bin"
            else:
                feat_file = n[:-1] + "_ds_" + str(ds) + "_features.bin"
                #acc_file = n[:-1] + "_ds_" + str(ds) + "_acc.bin"
                label_file = n[:-1] + "_ds_" + str(ds) + "_labels.bin"

            labels = np.memmap(label_file, dtype='ubyte', mode='r')

            # print(len(labels), np.count_nonzero(labels==1)) # check label split

            idxs = [i for i in range(0, labels.shape[0])]

            if len(idxs) == 0:
                continue

            for j in idxs:
                #data_tuple = (cam_file, acc_file, j, labels[j])
                data_tuple = (feat_file, j, labels[j])
                self.data_tuples.append(data_tuple)

        #shuffle(self.data_tuples)

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_tuple = self.data_tuples[idx]

        feat_sample = np.memmap(sample_tuple[0], dtype='float32', mode='r+', offset=4*sample_tuple[1]*self.features_offset, shape=self.features_shape)

        label = float(sample_tuple[2])
        # label = sample_tuple[2]

        # X = torch.stack((depth, vel, depth_conv, acc_conv, acc_conv2), dim=1).flatten()
        # X = torch.stack((depth, vel), dim=1).flatten()
        # return X, label
        return feat_sample, label