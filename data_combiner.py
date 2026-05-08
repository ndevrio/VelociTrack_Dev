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
import itertools
from funcs import combine_data

###############################
####    Data Load Section  ####
###############################

    
    
### [A] Single file run    
# combine_data(sys.argv[1], sys.argv[2])

### [B] Run all files
arg1_list = ["p20"]#["p1", "p2", "p3", "p4", "p5", "p6", "p8", "p9", "p10", "p11"]
arg2_list = range(1, 2)
comb_list = list(itertools.product(arg1_list, arg2_list))

if(len(sys.argv) <= 1):
    ds = 1
else:
    ds = int(sys.argv[1])

for i in tqdm(range(len(comb_list))):
    print(comb_list)
    print(comb_list[i][0], str(comb_list[i][1]))
    cam_win, acc_win, gt, _ = combine_data(comb_list[i][0], str(comb_list[i][1]), ds)
    
    ###########################################
    ###  STEP 1: Initial reshaping of data  ###
    ###########################################

    ### Create windowed structured data
    labels_win = gt.astype(int)
    labels_win = labels_win - np.roll(labels_win, 1) #Adjust format
    
    ######################################################################
    ###  STEP 2: Preprocessing and featurization function definitions  ###
    ######################################################################
    window_size = int(320/ds) # used to calculate overlap (320 / 800 = 400 ms)
    m = int(window_size / 10) # 10% overlap
    num_windows = int((labels_win.shape[0] - window_size - 1) / m)
    labels = np.zeros((num_windows, window_size))
    cam = np.zeros((num_windows, window_size, 8))
    acc = np.zeros((num_windows, window_size))


    def preprocessing():
        global labels, cam, acc
        
        def normalize_depth(data):
            mu = np.mean(data)
            
            data -= mu
            data /= 25
            return data

        # [1] Split continuous data into windows with 20% overlap
        for k in range(num_windows):
            labels[k] = labels_win[(k*m):(k*m)+window_size]
            cam[k] = cam_win[(k*m):(k*m)+window_size]
            acc[k] = acc_win[(k*m):(k*m)+window_size]
            
            #for i in range(cam.shape[2]-2):
            #    cam[k, :, i] = normalize(cam[k, :, i])
            
            cam[k, :, 0] = normalize_depth(cam[k, :, 0]) # normalize depth
            # acc[k] = normalize(acc[k])

        # [2] Reformat labels to classification labels for the entire window
        #q = int(labels.shape[1] / 4)
        #labels_middle_half = labels[:, q:3*q]
        #labels_adj = labels_adj - np.roll(labels_adj, 1, axis=1)
        #labels_adj[:, 1:-1] # fix edge issues

        #for i in range(len(labels_middle_half)):
        #    print(labels_middle_half[i, :1000])
        #exit()

        # print(labels.shape)
        # exit()
        # labels_a = np.any(labels[:, int(window_size/4):-int(window_size/4)]==1, axis=1).astype(int)
        
        # labels_a = np.any(labels[:, :int(window_size/2)]==1, axis=1).astype(int) # if there is a label transition in the first half of the window, mark that window as a true positive
        ### Change to first quarter instead of first half when making the window size larger
        labels_a = np.any(labels[:, :int(window_size/4)]==1, axis=1).astype(int)
        
        # labels_a = np.any(labels==1, axis=1).astype(int)
        
        # labels_b = np.sum(labels==-1, axis=1).astype(int) * -1

        labels = labels_a #+ labels_b
        # labels[labels > 1] = 1
        # print(labels[1014], labels_new[1014], cam[1014, :, 0])
        #print(labels_new[3886], labels_a[3886], labels_b[3886])
        #print(labels.shape)
        #print(labels[3884])

        # Get rid of windows where valid is true
        select = ~np.any(cam[:, :, -1]==1, axis=1)
        
        cam = cam[select]
        labels = labels[select]
        acc = acc[select]


    preprocessing()

    # Remove potentially bad data at the beginning and end of each recording
    cutoff_size = 10
    labels = labels[cutoff_size:-cutoff_size]
    cam = cam[cutoff_size:-cutoff_size]
    acc = acc[cutoff_size:-cutoff_size]

    ### Write out to memap file
    outfile = "./study_data/" + comb_list[i][0] + "/" + str(comb_list[i][1])

    if(ds != 1):
        outfile += "_ds_" + str(ds)  
    
    labels_fp = np.memmap(outfile + "_labels" + ".bin", dtype='ubyte', mode='w+', shape=labels.shape)
    labels_fp[:] = labels[:]
    labels_fp.flush()
    print('Wrote\t', outfile + "_labels" + ".bin\t", labels_fp.shape)

    cam_fp = np.memmap(outfile + "_cam" + ".bin", dtype='float32', mode='w+', shape=cam.shape)
    cam_fp[:] = cam[:]
    cam_fp.flush()
    print('Wrote\t', outfile + "_cam" + ".bin\t", cam_fp.shape)

    acc_fp = np.memmap(outfile + "_acc" + ".bin", dtype='float32', mode='w+', shape=acc.shape)
    acc_fp[:] = acc[:]
    acc_fp.flush()
    print('Wrote\t', outfile + "_acc" + ".bin\t", acc_fp.shape)

    