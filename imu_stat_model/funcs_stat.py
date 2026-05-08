import pandas as pd
import numpy as np

def combine_data(arg1, arg2, ds=1):
    ### Data descriptions:
    ### [Timestamp] [Finger X] [Finger Y] [Mediapipe reset] [Unity near wall]
    ### [Finger X] [Finger Y] [Mediapipe reset] [Unity near wall]
    ### [IMU X] [IMU Y] [IMU Z] [Pressure]
    # imu_df = pd.read_csv("..\\study_data\\" + arg1 + "\\imu_data_" + arg2 + ".txt", sep=',', header=None)

    # Merge three files into one
    features = np.load("../study_data/" + arg1 + "/" + arg2 + "_viz_data.npy")
    features = features[:, -5:-1]

    # Downsample for ablation study, if used
    features = features[::ds]

    gt = np.load("..\\study_data\\" + arg1 + "\\gt_data_" + arg2 + ".npy")[::ds]


    shift_idx = int(arg1[1:])-1
    imu_shifts = [10, 10, 10, 5, 10, 10, 10, -10, -10, 10, 10, 10]
    
    if(shift_idx > len(imu_shifts)):
        shift_idx = 0

    ###########################
    ###   Feature creation  ###
    ###########################

    ### [1] Shift amount for time difference between IMU and camera data
    gt = np.roll(gt, int(imu_shifts[shift_idx]/ds), axis=0)
    
    ### [2] Standardize across each data file
    imu_features = features[:, -4:]#:-1]
    # imu_features[:, :3] = (imu_features[:, :3] - np.mean(imu_features[:, :3], axis=0))# / np.std(imu_features[:, :3], axis=0)

    ###########################
    ###   Feature creation  ###
    ###########################
    
    return imu_features, gt


def create_stat_features(acc):
    from scipy.stats import skew, kurtosis
    
    # stat_features = np.zeros((len(acc), 5*3))

    # # [0] Maximum
    # stat_features[:, 0:3] = np.max(acc, axis=1)
    # # [1] Minimum
    # stat_features[:, 3:6] = np.min(acc, axis=1)
    # # [2] Mean
    # stat_features[:, 6:9] = np.mean(acc, axis=1)
    # # [3] Minimum
    # stat_features[:, 9:12] = skew(acc, axis=1)
    # # [4] Minimum
    # stat_features[:, 12:15] = kurtosis(acc, axis=1)

    stat_features = np.zeros((len(acc), 1))
    
    #np.linalg.norm(acc, axis=2)

    # [0] Maximum
    stat_features[:, 0] = np.max(acc, axis=1) - np.min(acc, axis=1)
    # [1] Minimum
    # stat_features[:, 1] = np.min(acc, axis=1)
    # # [2] Mean
    # stat_features[:, 2] = np.std(acc, axis=1)
    # # [3] Minimum
    # stat_features[:, 3] = skew(acc, axis=1)
    # # [4] Minimum
    # stat_features[:, 4] = kurtosis(acc, axis=1)

    return stat_features