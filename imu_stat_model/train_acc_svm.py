import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import sys
import os
import time
import pandas as pd
from tqdm import tqdm
import joblib

from funcs_stat import create_stat_features

np.set_printoptions(precision=4)


def get_training_data(data_dir, participant, trial_mode, test_mode):

    files = []
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
                files.append(line)
            elif(test_mode and check):
                files.append(line)


    labels_list = []
    acc_list = []
    for n in files:
        acc_file = n[:-1] + "_stat_acc.bin"
        label_file = n[:-1] + "_stat_labels.bin"

        labels = np.memmap(label_file, dtype='ubyte', mode='r')

        acc_shape = (len(labels), 40, 4)
        acc = np.memmap(acc_file, dtype='float32', mode='r+', shape=acc_shape)[:, :, :-1]

        labels_list.append(labels)
        acc_list.append(acc)

    #shuffle(data_tuples)

    labels = np.concatenate(labels_list, axis=0)
    acc = np.concatenate(acc_list, axis=0)

    print(np.sum(labels==1), len(labels))

    ### Featurization
    stat_features = create_stat_features(acc)

    idxs = np.random.permutation(len(stat_features))

    stat_features = stat_features[idxs]
    labels = labels[idxs]

    return stat_features, labels


def main():
    X_train, y_train = get_training_data(sys.argv[1], sys.argv[2], sys.argv[3], 0)
    X_test, y_test  = get_training_data(sys.argv[1], sys.argv[2], sys.argv[3], 1)

    # X_train[y_train == 1] = 1
    # X_test[y_test == 1] = 1

    ########################
    ###  Model training  ###
    ########################
    # clf = RandomForestClassifier(n_estimators=500, max_depth=5, max_features=2, class_weight={0: 0.05, 1:0.95})
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight={0: 0.05, 1:0.95})
    # clf = SVC(class_weight={0: 0.05, 1:0.95})

    print(X_train.shape, y_train.shape)

    print("Creating sample weights...")
    sample_weight = np.array([0.05 if i == 0 else 0.95 for i in y_train])

    print("Fitting...")
    clf.fit(X_train, y_train)#, sample_weight=sample_weight)

    ##########################
    ###  Model evaluation  ###
    ##########################
    print("Predicting...")
    y_pred = clf.predict(X_test)

    cmat = confusion_matrix(y_test, y_pred)

    precision = cmat[1, 1] / (cmat[1, 1] + cmat[0, 1])
    recall = cmat[1, 1] / (cmat[1, 1] + cmat[1, 0])
    print(cmat, precision, recall)

    np.save((sys.argv[2] + "_" + sys.argv[3] + "_ypred.npy"), y_pred)
    
    print("Saving model to file...")
    modle_filename = "statmodel_" + sys.argv[2] + "_" + sys.argv[3] + ".joblib"
    joblib.dump(clf, modle_filename)


if __name__ == "__main__":
    main()



