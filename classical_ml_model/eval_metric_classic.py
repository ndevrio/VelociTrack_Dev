import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import sys
from tqdm import tqdm

import torch
from mmapdataset import RealSenseDataset
from models_new import Depth1DCNN, TimeSeriesData
from funcs import combine_data

import joblib


# Downsample factor for ablation study
if(len(sys.argv) <= 1):
    ds = 1
else:
    ds = int(sys.argv[1])


def run_file(arg1, arg2, model_idx):
    ###############################
    ####    Data Load Section  ####
    ###############################
    label_file = "./study_data/" + arg1 + "/" + str(arg2) + "_labels.bin"
    features_file = "./study_data/" + arg1 + "/" + str(arg2) + "_features.bin"

    gt = np.memmap(label_file, dtype='ubyte', mode='r')
    features = np.memmap(features_file, dtype='float32', mode='r+')#, shape=self.features_shape)
    features = features.reshape(-1, 1)

    ############################
    ####    Model Section   ####
    ############################
    model = joblib.load('rf_model_p1_0.joblib')


    #################################
    ####    Create predictions   ####
    #################################
    predictions = np.zeros(gt.shape)

    for i in tqdm(range(len(features))):
        result = model.predict(features[i].reshape(1, -1))
        result = result == gt[i]        
        predictions[i] = result.astype(int)

    state = 0
    last_pred = 0
    last_gt = 0
    c_mat = np.zeros((2, 2))

    for i in range(len(predictions)):
        pred = predictions[i]
        g = gt[i]
        if pred == 1 and last_pred == 0:
            if(state == 0):
                state = 1
            if(state == 2):
                state = 0
                c_mat[1, 1] += 1 # tp

        if g == 1 and last_gt == 0:
            if(state == 0):
                state = 2
            if(state == 1):
                state = 0
                c_mat[1, 1] += 1 # tp

        if pred == 0 and last_pred == 1:
            if(state == 1):
                state = 0
                c_mat[0, 1] += 1 # fp

        if g == 0 and last_gt == 1:
            if(state == 2):
                state = 0
                c_mat[1, 0] += 1 # fn
        
        last_pred = pred
        last_gt = g

    # c_mat = confusion_matrix(gt, predictions)

    return c_mat


participants = ["p3"]#, "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p10", "p11"]
with open('output_metric_classic.txt', 'a') as f:
    f.write("Results: " + str(ds) + "\n")
    for p in range(len(participants)):
        c_mat = np.zeros((2, 2))
        for n_1 in range(20):
            n = n_1

            c_res = run_file(participants[p], n+1, p)
            c_mat += c_res

            f.write(participants[p] + "," + str(n) + "," + str(int(c_res[0, 0])) + "," + str(int(c_res[0, 1])) + "," + str(int(c_res[1, 0])) + "," + str(int(c_res[1, 1])) + "\n")
            f.flush()

        f.write("All taps:  " + participants[p] + "," + str(int(c_mat[0, 0])) + "," + str(int(c_mat[0, 1])) + "," + str(int(c_mat[1, 0])) + "," + str(int(c_mat[1, 1])) + "\n" + "\n")
        f.flush()

