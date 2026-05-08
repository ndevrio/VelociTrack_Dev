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
import joblib

from funcs_stat import create_stat_features


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
        infile = "..\\study_data\\" + arg1 + "\\" + str(arg2) + "_viz_data.npy"
    else:
        infile = "..\\study_data\\" + arg1 + "\\" + str(arg2) + "_ds_" + str(ds) + "_viz_data.npy"
    data = np.load(infile, allow_pickle=True)

    infile_valid = "../study_data/" + arg1 + "/valid_data_" + str(arg2) + ".npy"
    valid = np.load(infile_valid, allow_pickle=True)
    gt = np.load("..\\study_data\\" + arg1 + "\\gt_data_" + str(arg2) + ".npy")[::ds]
    acc = data[:, -5:-2]

    acc = acc - np.mean(acc, axis=0) 
    std = np.std(acc, axis=0)
    std[std == 0] = 1

    # acc = acc / 0.25#std
    acc = acc[:, 0]

    ### Correct IMU data
    shift_idx = int(arg1[1:])-1
    imu_shifts = [-50, -50, -50, -50, -50, -50, -50, -10, -10, -50, -50, -50]
    if(shift_idx > len(imu_shifts)):
        shift_idx = 0
    gt = np.roll(gt, int(imu_shifts[shift_idx]/ds), axis=0)

    gt_mod = gt - np.roll(gt, 1) #Adjust format
    select = np.ones(gt.shape, dtype=np.int32)
    cnt = 0
    for i in range(len(gt_mod)-1):
        if(gt_mod[i] == -1):
            cnt = 200
        if(cnt > 0):
            cnt -= 1
            select[i+1] = 0

    # print(gt[1000:2000].astype(np.int32))
    # print(select[1000:2000])

    # predictions = np.load("p4_0_ypred.npy")

    ############################
    ####    Model Section   ####
    ############################
    # clf = joblib.load(saved_models[model_idx][:-1])

    #################################
    ####    Create predictions   ####
    #################################
    window_size = int(10/ds)

    predictions = np.zeros((len(acc),))
    batch_size = 4096
    c = 1

    stat_in = np.zeros((batch_size, 1))

    buffer = int(1024/ds)
    for i in tqdm(range(buffer, len(acc)-buffer)):
        if(i >= int(window_size/2)):
            d = acc[i-int(window_size/2):i+int(window_size/2)]
            stat_features = create_stat_features(np.expand_dims(d, axis=0))

            # stat_in[c % batch_size] = stat_features

            # if((c % batch_size) == 0):
            #     # print(stat_in[:, 0])
            #     result = stat_in[:, 0] > 2.5#clf.predict(stat_in)
            #     predictions[i-batch_size:i] = result.astype(int)
            # elif(i == len(acc)-buffer-1):
            #     result = stat_in[:, 0] > 2.5#clf.predict(stat_in)
            #     predictions[i-(c % batch_size):i] = result.astype(int)[:c % batch_size]

            # c += 1
            predictions[i] = stat_features[0] > 0.75 #2.5

    # stat_features = create_stat_features(acc)
    # predictions = stat_features > 2.5
    predictions = predictions * select * (1-valid)

    np.save("my_preds.npy", predictions)


    ### Smooth out labels spurious by one
    last_p = -1
    last_p2 = -1
    for p in range(2, len(predictions)):
        if(predictions[p] == 0 and last_p == 1 and last_p2 == 0):
            predictions[p-1] = 0
        elif(predictions[p] == 1 and last_p == 0 and last_p2 == 1):
            predictions[p-1] = 1
        last_p2 = last_p
        last_p = predictions[p]

    # predictions = gt.copy()

    ### Go through predictions and turn window into 1 if >80 of prediction is 1
    mod_pred = np.zeros(predictions.shape)
    window_size = int(250/ds)
    t = 0.01
    
    state = 0
    last_pred = 0
    last_gt = 0
    mag_level = 0
    c_mat = np.zeros((3, 2, 2))

    for i in range(len(predictions)):
        if(i > int(window_size / 2) and i < (len(predictions) - int(window_size / 2) - 1)):
            s = np.sum(predictions[i-int(window_size/2):i+int(window_size/2)])
            mod_pred[i] = int(s > t*window_size)

            pred = mod_pred[i]
            g = gt[i]
            if pred == 1 and last_pred == 0:
                if(state == 0):
                    state = 1
                elif(state == 2):
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
                elif(state == 1):
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

    return c_mat


participants = ["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p10", "p11"]
with open('output_metric_stat.txt', 'a') as f:
    f.write("Results: " + str(ds) + "\n")
    for p in range(len(participants)):
        c_mat = np.zeros((3, 2, 2))
        for n_1 in range(20): # 20
            # n = 2*(n_1+1)
            n = n_1+1

            # Skip missing IMU data
            if(participants[p] == "p1" and n < 7):
                continue

            # print(participants[p], n+1, p)

            c_res = run_file(participants[p], n, int(participants[p][1:])-1)
            c_mat += c_res

            c_res = np.sum(c_res, axis=0)
            f.write(participants[p] + "," + str(n) + "," + str(int(c_res[0, 0])) + "," + str(int(c_res[0, 1])) + "," + str(int(c_res[1, 0])) + "," + str(int(c_res[1, 1])) + "\n")
            f.flush()

        f.write("Soft taps: " + participants[p] + "," + str(int(c_mat[0, 0, 0])) + "," + str(int(c_mat[0, 0, 1])) + "," + str(int(c_mat[0, 1, 0])) + "," + str(int(c_mat[0, 1, 1])) + "\n")
        f.write("Med. taps: " + participants[p] + "," + str(int(c_mat[1, 0, 0])) + "," + str(int(c_mat[1, 0, 1])) + "," + str(int(c_mat[1, 1, 0])) + "," + str(int(c_mat[1, 1, 1])) + "\n")
        f.write("Hard taps: " + participants[p] + "," + str(int(c_mat[2, 0, 0])) + "," + str(int(c_mat[2, 0, 1])) + "," + str(int(c_mat[2, 1, 0])) + "," + str(int(c_mat[2, 1, 1])) + "\n")
        c_mat = np.sum(c_mat, axis=0)
        f.write("All taps:  " + participants[p] + "," + str(int(c_mat[0, 0])) + "," + str(int(c_mat[0, 1])) + "," + str(int(c_mat[1, 0])) + "," + str(int(c_mat[1, 1])) + "\n")
        f.write("Precision: " + str(c_mat[1, 1] / (c_mat[1, 1] + c_mat[0, 1])) + "   Recall: " + str(c_mat[1, 1] / (c_mat[1, 1] + c_mat[1, 0])) + "\n" + "\n")
        f.flush()

