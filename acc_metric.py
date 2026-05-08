import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
import sys
from tqdm import tqdm

import torch
from mmapdataset import RealSenseDataset
from models_new import Acc1DCNN, TimeSeriesData
from funcs import combine_data


with open('saved_models.txt', 'r') as file:
    saved_models = file.readlines()


def run_file(arg1, arg2, model_idx):
    ###############################
    ####    Data Load Section  ####
    ###############################
    infile = "./study_data/" + arg1 + "/" + str(arg2) + "_viz_data.npy"
    data = np.load(infile, allow_pickle=True)

    gt = data[:, -1]
    valid = data[:, 11]
    acc = data[:, -5:-2]

    imu_mag = np.sqrt(np.exp2(data[:, -5]) + np.exp2(data[:, -4]) + np.exp2(data[:, -3]))
    imu_mag /= 5

    ############################
    ####    Model Section   ####
    ############################
    model = Acc1DCNN()
    model.load_state_dict(torch.load(saved_models[model_idx][:-1])["state_dict"])
    model.zero_grad()
    model.eval()
    model.to('cuda')


    #################################
    ####    Create predictions   ####
    #################################

    window_size = 320
    predictions = np.zeros((len(imu_mag),))
    batch_size = 512

    c = 1

    imu_in = torch.zeros((batch_size, window_size, 1)).to('cuda')


    buffer = 1024
    for i in tqdm(range(buffer, len(imu_mag)-buffer)):
        if(i >= int(window_size/2)):
            imu_in[c % batch_size] = torch.from_numpy(imu_mag[i-int(window_size/2):i+int(window_size/2)]).float().unsqueeze(1).to('cuda')
            
            if((c % batch_size) == 0):
                # X = torch.stack((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in), dim=2)
                # X = X.flatten(start_dim=1)
                # result = model(X)[:, 0]

                result = model(imu_in)[:, 0]

                result = (torch.sigmoid(result) >= 0.5).cpu().numpy()
                # result = model((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in))
                # result = torch.argmax(result, 1).cpu().numpy()
                
                predictions[i-batch_size:i] = result.astype(int)
                # print(predictions[i])
            
            c += 1


    ### Go through predictions and turn window into 1 if >80 of prediction is 1
    mod_pred = np.zeros(predictions.shape)
    window_size = 96
    t = 0.05

    state = 0
    last_pred = 0
    last_gt = 0
    mag_level = 0
    c_mat = np.zeros((3, 2, 2))

    for i in range(len(predictions)):
        if(i > int(window_size / 2) and i < (len(predictions) - int(window_size / 2) - 1) and valid[i] == 0):
            s = np.sum(predictions[i-int(window_size/2):i+int(window_size/2)])
            mod_pred[i] = int(s > t*window_size)

            pred = mod_pred[i]
            g = gt[i]
            if pred == 1 and last_pred == 0:
                if(state == 0):
                    state = 1
                if(state == 2):
                    state = 0
                    c_mat[mag_level, 1, 1] += 1 # tp

            if g == 1 and last_gt == 0:
                # Find acc_mag for this ground truth
                acc_mag = np.max(acc[i-int(window_size/2):i+int(window_size/2)])*5
                if(acc_mag <= 5.0):
                    mag_level = 0   # soft tap
                elif(acc_mag > 5.0 and acc_mag <= 15.0):
                    mag_level = 1   # medium tap
                else:
                    mag_level = 2   # hard tap

                if(state == 0):
                    state = 2
                if(state == 1):
                    state = 0
                    c_mat[mag_level, 1, 1] += 1 # tp

            if pred == 0 and last_pred == 1:
                if(state == 1):
                    state = 0
                    c_mat[mag_level, 0, 1] += 1 # fp

            if g == 0 and last_gt == 1:
                if(state == 2):
                    state = 0
                    c_mat[mag_level, 1, 0] += 1 # fn
            
            last_pred = pred
            last_gt = g
        elif(valid[i] == 1):
            state = 0

    return c_mat


participants = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p10", "p11"]
with open('output_metric.txt', 'a') as f:
    for p in range(len(participants)):
        c_mat = np.zeros((3, 2, 2))
        print(saved_models[p][:-1])
        for n_1 in range(10): # 20
            if(participants[p] == "p1" and n_1 < 3): # Skip files with missing IMU data
                continue

            n = (2*n_1)+1
            # n = n_1

            c_res = run_file(participants[p], n+1, p)
            c_mat += c_res

            c_res = np.sum(c_res, axis=0)
            f.write(participants[p] + "," + str(n) + "," + str(int(c_res[0, 0])) + "," + str(int(c_res[0, 1])) + "," + str(int(c_res[1, 0])) + "," + str(int(c_res[1, 1])) + "\n")
            f.flush()

        f.write("Soft taps: " + participants[p] + "," + str(int(c_mat[0, 0, 0])) + "," + str(int(c_mat[0, 0, 1])) + "," + str(int(c_mat[0, 1, 0])) + "," + str(int(c_mat[0, 1, 1])) + "\n")
        f.write("Med. taps: " + participants[p] + "," + str(int(c_mat[1, 0, 0])) + "," + str(int(c_mat[1, 0, 1])) + "," + str(int(c_mat[1, 1, 0])) + "," + str(int(c_mat[1, 1, 1])) + "\n")
        f.write("Hard taps: " + participants[p] + "," + str(int(c_mat[2, 0, 0])) + "," + str(int(c_mat[2, 0, 1])) + "," + str(int(c_mat[2, 1, 0])) + "," + str(int(c_mat[2, 1, 1])) + "\n")
        c_mat = np.sum(c_mat, axis=0)
        f.write("All taps:  " + participants[p] + "," + str(int(c_mat[0, 0])) + "," + str(int(c_mat[0, 1])) + "," + str(int(c_mat[1, 0])) + "," + str(int(c_mat[1, 1])) + "\n" + "\n")
        f.flush()

