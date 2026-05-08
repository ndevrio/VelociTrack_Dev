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
from models_new import Depth1DCNN, TimeSeriesData
from funcs import combine_data


# Downsample factor for ablation study
if(len(sys.argv) <= 1):
    ds = 1
    model_file = "saved_models.txt"
else:
    ds = int(sys.argv[1])
    model_file = "saved_models_ds_" + str(ds) + ".txt"

with open(model_file, 'r') as file:
    saved_models = file.readlines()


def run_file(arg1, arg2, model_idx):
    ###############################
    ####    Data Load Section  ####
    ###############################
    if(ds == 1):
        infile = "./study_data/" + arg1 + "/" + str(arg2) + "_viz_data.npy"
    else:
        infile = "./study_data/" + arg1 + "/" + str(arg2) + "_ds_" + str(ds) + "_viz_data.npy"
    data = np.load(infile, allow_pickle=True)

    cam = data[:, 5:11]
    gt = data[:, -1]
    valid = data[:, 11]
    acc = data[:, -5:-2]

    ############################
    ####    Model Section   ####
    ############################
    model = Depth1DCNN(ds)
    model.load_state_dict(torch.load(saved_models[model_idx][:-1])["state_dict"])
    model.zero_grad()
    model.eval()
    model.to('cuda')


    #################################
    ####    Create predictions   ####
    #################################

    window_size = int(320/ds)
    predictions = np.zeros((len(cam),))
    batch_size = 512

    c = 1

    # cam_shape = (320, 8)
    # cam_offset = cam_shape[0]*cam_shape[1]
    
    # cam_sample = np.memmap("./study_data" + arg1 + "/" + arg2 + "_cam.bin", dtype='float32', mode='r+', offset=4*idx*cam_offset, shape=cam_shape)
    # cam_sample = torch.as_tensor(cam_sample)

    depth_in = torch.zeros((batch_size, window_size, 1)).to('cuda')
    vel_in = torch.zeros((batch_size, window_size, 1)).to('cuda')
    conv_depth_in = torch.zeros((batch_size, window_size, 1)).to('cuda')
    conv_vel_in = torch.zeros((batch_size, window_size, 1)).to('cuda')
    conv_vel2_in = torch.zeros((batch_size, window_size, 1)).to('cuda')

    def normalize_depth(data):
        mu = np.mean(data)
        
        data -= mu
        data /= 25
        return data

    buffer = int(1024/ds)
    for i in tqdm(range(buffer, len(cam)-buffer)):
        if(i >= int(window_size/2)):
            d = cam[i-int(window_size/2):i+int(window_size/2), 0].copy()
            d = normalize_depth(d)
            depth_in[c % batch_size] = torch.from_numpy(d).float().unsqueeze(1).to('cuda')
            vel_in[c % batch_size] = torch.from_numpy(cam[i-int(window_size/2):i+int(window_size/2), 1]).float().unsqueeze(1).to('cuda')
            
            conv_depth_in[c % batch_size] = torch.from_numpy(cam[i-int(window_size/2):i+int(window_size/2), 3]).float().unsqueeze(1).to('cuda')
            conv_vel_in[c % batch_size] = torch.from_numpy(cam[i-int(window_size/2):i+int(window_size/2), 4]).float().unsqueeze(1).to('cuda')
            conv_vel2_in[c % batch_size] = torch.from_numpy(cam[i-int(window_size/2):i+int(window_size/2), 5]).float().unsqueeze(1).to('cuda')
            
            if((c % batch_size) == 0):
                # X = torch.stack((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in), dim=2)
                # X = X.flatten(start_dim=1)
                # result = model(X)[:, 0]

                result = model((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in))[:, 0]

                result = (torch.sigmoid(result) >= 0.5).cpu().numpy()
                # result = model((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in))
                # result = torch.argmax(result, 1).cpu().numpy()
                
                predictions[i-batch_size:i] = result.astype(int)
                # print(predictions[i])
            
            c += 1


    ### Go through predictions and turn window into 1 if >80 of prediction is 1
    mod_pred = np.zeros(predictions.shape)
    window_size = int(96/ds)
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
with open('output_metric_ablation.txt', 'a') as f:
    f.write("Results: " + str(ds) + "\n")
    for p in range(len(participants)):
        c_mat = np.zeros((3, 2, 2))
        print(saved_models[p][:-1])
        for n_1 in range(10): # 20
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

