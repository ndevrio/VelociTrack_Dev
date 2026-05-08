import pandas as pd
import numpy as np

def combine_data(arg1, arg2, ds=1):
    has_imu = True

    ### Data descriptions:
    ### [Timestamp] [Finger X] [Finger Y] [Mediapipe reset] [Unity near wall]
    ### [Finger X] [Finger Y] [Mediapipe reset] [Unity near wall]
    ### [IMU X] [IMU Y] [IMU Z] [Pressure]
    cam1_df = pd.read_csv(".\\study_data\\" + arg1 + "\\cam1_data_" + arg2 + ".txt", sep=',', header=None)
    cam2_df = pd.read_csv(".\\study_data\\" + arg1 + "\\cam2_data_" + arg2 + ".txt", sep=',', header=None)
    try:
        imu_df = pd.read_csv(".\\study_data\\" + arg1 + "\\imu_data_" + arg2 + ".txt", sep=',', header=None)
    except:
        has_imu = False

    # Merge three files into one
    features = pd.merge_asof(cam1_df, cam2_df, 0, direction="nearest")
    if(has_imu):
        features = pd.merge_asof(features, imu_df, 0, direction="nearest")
    features = features.astype(float).to_numpy()
    if(not has_imu):
        imu_blank = np.zeros((len(features), 4))
        features = np.concatenate((features, imu_blank), axis=1)
    else:
        features = features[:, :-1] # remove bad feature at the back

    # Downsample for ablation study, if used
    features = features[::ds]

    mp_did_reset = np.logical_or(features[:, 3], features[:, 8]).astype(np.int16)
    mp_hand_tracked = np.logical_and(features[:, 4], features[:, 9]).astype(np.int16)

    gt = np.load(".\\study_data\\" + arg1 + "\\gt_data_" + arg2 + ".npy")[::ds]
    valid = np.load(".\\study_data\\" + arg1 + "\\valid_data_" + arg2 + ".npy")[::ds]


    shift_idx = int(arg1[1:])
    imu_shifts = [50, 50, 50, 50, 50, 50, 50, 35, 35, 50, 50, 50]
    unity_shifts = [0, 0, 0, 0, -100, -75, -70, -50, -50, -35, -35, -35]
    
    if(shift_idx > len(imu_shifts)):
        shift_idx = 0

    ###########################
    ###   Feature creation  ###
    ###########################

    ### [1] Shift amount for time difference between IMU and camera data
    print(shift_idx, features.shape)
    features[:, -4:] = np.roll(features[:, -4:], int(imu_shifts[shift_idx]/ds), axis=0)


    ### [2] Shift amount for time difference between Unity data and camera data
    near_wall = np.roll(features[:, 5], int(unity_shifts[shift_idx]/ds), axis=0)

    ### [3] Convert Y values from each camera into a single depth value using camera geometry
    diff = features[:, 2].copy() - features[:, 7].copy()

    # Correct for any divide by zero error
    for i in range(len(diff)):
        if diff[i] == 0:
            diff[i] = diff[i-1]

    # Account for flipped values in first three participants
    if(arg1 == "p1" or (arg1 == "p2" and int(arg2) != 3 and int(arg2) != 4) or arg1 == "p3" or arg1 == "p20"):
        depth = -diff
    else:
        depth = diff
    
    ### [4] Filter the depth data using a OneEuroFilter
    import math
    """class OneEuroFilter:
        def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
            self.min_cutoff = min_cutoff
            self.beta = beta
            self.d_cutoff = d_cutoff
            self.x_prev = 0.0
            self.dx_prev = 0.0
            self.t_prev = 0.0

        def filter(self, x, t):
            if self.t_prev == 0.0:
                self.t_prev = t
                self.x_prev = x
                return x

            dt = t - self.t_prev
            if(dt == 0):
                return self.x_prev
            dx = (x - self.x_prev) / dt

            ed = math.exp(-self.d_cutoff * dt)
            dx_hat = self.dx_prev * ed + (1.0 - ed) * dx

            cutoff = self.min_cutoff + self.beta * abs(dx_hat)
            e = math.exp(-cutoff * dt)
            x_hat = self.x_prev * e + (1.0 - e) * x

            self.t_prev = t
            self.x_prev = x_hat
            self.dx_prev = dx_hat

            return x_hat
    one_euro_filter = OneEuroFilter(min_cutoff=1.5, beta=0.5)

    # window_size = 50
    # flat_window = np.zeros((window_size,))
    # new_features = np.zeros((len(features), 1))
    start_time = features[0, 0]
    for i in range(len(features)):
        # flat_window = np.roll(flat_window, 1)
        # flat_window[0] = depth[i]
        
        # [A] OneEuroFilter data filtering
        if not np.isinf(depth[i]):
            if(i == 0):
                depth[i] = one_euro_filter.filter(depth[i], 0)
            else:
                depth[i] = one_euro_filter.filter(depth[i], features[i, 0]-start_time)
        
        # [B] Custom features
        # if(i >= window_size): # If the window is valid
            ### [1] Finger is not shaking for X s within the window
            ### STD of window is < Z
            # new_features[i, 0] = int(np.abs(flat_window[-1] - flat_window[0]) < 3.0)
            # if(np.abs(flat_window[-1] - flat_window[0]) < 3.0):
            #     new_features[i, 0] = 1
            # else:
            #     new_features[i, 0] = 0
            # new_features[i, 0] = int(np.median(flat_window) == 0)"""

    ### [5] Derive velocity and acceleration data from depth
    r_val = int(16/ds)
    if(r_val < 1):
        r_val = 1
    velocity = depth - np.roll(depth, r_val)
    velocity = np.roll(velocity, -r_val)
    
    velocity /= 10
    
    acc = velocity - np.roll(velocity, r_val)
    acc = np.roll(acc, -r_val)
    
    ### [6] Get the magnitude squared of the X, Y, and Z accelerometer values
    imu_mag = np.sqrt(np.exp2(features[:, -4]) + np.exp2(features[:, -3]) + np.exp2(features[:, -2]))
    imu_mag /= 5 # scale into ~0-1 range
    
    
    ### [7] Generate convolution-based features
    convolve_features = generate_convolve_features(depth, velocity, ds)
    
    ### [8] Normalize data streams    
    def normalize(data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    
    # May need to adjust these based on "valid" data
    # depth = normalize(depth)
    # velocity = normalize(velocity)
    # acc = normalize(acc)
    # imu_mag = normalize(imu_mag)
    

    ### Save data format:
    ### 0-7 Features:                       [Timestamp] [Cam 1 Finger X] [Cam 1 Finger Y] [Cam 2 Finger X] [Cam 2 Finger Y] [Depth] [Velocity] [Acc] [depth_conv] [acc_conv] [acc_conv2]
    ### 8-11 Data validity:                 [valid] [mp_did_reset] [mp_hand_tracked] [near_wall] 
    ### 12-15 IMU data:                     [IMU X] [IMU Y] [IMU Z] [Pressure]
    ### 16 Ground truth                     [gt]
    output_save = np.column_stack((features[:, :3], features[:, 6:8], depth, velocity, acc, convolve_features, valid, mp_did_reset, mp_hand_tracked, near_wall, features[:, -4:], gt))
    if(ds == 1):
        outfile = "./study_data/" + arg1 + "/" + arg2 + "_viz_data.npy"
    else:
        outfile = "./study_data/" + arg1 + "/" + arg2 + "_ds_" + str(ds) + "_viz_data.npy"
    np.save(outfile, output_save)

    ### Model data format:                  [Depth] [Velocity] [Acc] [Depth conv feature] [Acc conv feature] [Acc conv feature 2] [near_wall] [valid]
    output_model = np.column_stack((depth, velocity, acc, convolve_features, near_wall, valid))

    return output_model, imu_mag, gt, valid




def generate_convolve_features(depth, vel, ds=1):
    convolve_window_size = int(80/ds)
    # depth_ex_signal = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    # acc_ex_signal = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0,-0.05,-0.1,-0.15,-0.2,-0.25,-0.3,-0.35,-0.4,-0.45,-0.5,-0.55,-0.6,-0.65,-0.7,-0.75,-0.8,-0.85,-0.9,-0.95,-1,-0.95,-0.9,-0.85,-0.8,-0.75,-0.7,-0.65,-0.6,-0.55,-0.5,-0.45,-0.4,-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    depth_ex_signal_2 = np.array([0.002182716,0.002472623,0.002800927,0.003172683,0.003593603,0.004070138,0.004609572,0.005220126,0.005911069,0.006692851,0.007577241,0.008577485,0.009708476,0.010986943,0.012431651,0.014063627,0.015906392,0.01798621,0.020332353,0.02297737,0.025957357,0.029312231,0.033085978,0.037326887,0.042087728,0.047425873,0.05340333,0.06008665,0.067546691,0.07585818,0.085099045,0.095349465,0.106690594,0.119202922,0.13296424,0.148047198,0.164516463,0.182425524,0.201813222,0.222700139,0.245085013,0.268941421,0.294214972,0.320821301,0.348645135,0.377540669,0.4073334,0.437823499,0.468790627,0.5,0.531209373,0.562176501,0.5926666,0.622459331,0.651354865,0.679178699,0.705785028,0.731058579,0.754914987,0.777299861,0.798186778,0.817574476,0.835483537,0.851952802,0.86703576,0.880797078,0.893309406,0.904650535,0.914900955,0.92414182,0.932453309,0.93991335,0.94659667,0.952574127,0.957912272,0.962673113,0.966914022,0.970687769,0.974042643,0.97702263,0.979667647,0.98201379,0.984093608,0.985936373,0.987568349,0.989013057,0.990291524,0.991422515,0.992422759,0.993307149,0.994088931,0.994779874,0.995390428,0.995929862,0.996406397,0.996827317,0.997199073,0.997527377,0.997817284,0.998073265,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    vel_ex_signal_2 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.02500,0.05000,0.07500,0.10000,0.12500,0.15000,0.17500,0.20000,0.22500,0.25000,0.27500,0.30000,0.32500,0.35000,0.37500,0.40000,0.42500,0.45000,0.47500,0.50000,0.52500,0.55000,0.57500,0.60000,0.62500,0.65000,0.67500,0.70000,0.72500,0.75000,0.77500,0.80000,0.82500,0.85000,0.87500,0.90000,0.92500,0.95000,0.97500,1.00000,0.97500,0.95000,0.92500,0.90000,0.87500,0.85000,0.82500,0.80000,0.77500,0.75000,0.72500,0.70000,0.67500,0.65000,0.62500,0.60000,0.57500,0.55000,0.52500,0.50000,0.47500,0.45000,0.42500,0.40000,0.37500,0.35000,0.32500,0.30000,0.27500,0.25000,0.22500,0.20000,0.17500,0.15000,0.12500,0.10000,0.07500,0.05000,0.02500,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000])

    depth_ex_signal_1 = np.array([0.051186454,0.053952699,0.056754637,0.059593204,0.062469368,0.06538414,0.06833857,0.071333752,0.074370826,0.07745098,0.080575455,0.083745544,0.086962599,0.090228032,0.093543322,0.096910013,0.100329725,0.103804155,0.107335082,0.110924375,0.114573994,0.118286003,0.122062572,0.125905986,0.129818655,0.13380312,0.137862065,0.141998328,0.146214912,0.150514998,0.15490196,0.159379381,0.163951071,0.168621084,0.173393743,0.178273662,0.183265772,0.188375355,0.193608072,0.198970004,0.204467696,0.210108202,0.215899138,0.22184875,0.227965978,0.234260541,0.24074303,0.247425011,0.254319153,0.261439373,0.268801001,0.276420984,0.284318118,0.292513326,0.301029996,0.309894379,0.319136082,0.32878866,0.338890353,0.349485002,0.3606232,0.372363747,0.384775539,0.397940009,0.41195437,0.426935982,0.443028324,0.460409377,0.479303657,0.5,0.522878745,0.548455007,0.57745098,0.610924375,0.650514998,0.698970004,0.761439373,0.849485002,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
    # vel_ex_signal_1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.016666667,-0.033333333,-0.05,-0.066666667,-0.083333333,-0.1,-0.116666667,-0.133333333,-0.15,-0.166666667,-0.183333333,-0.2,-0.216666667,-0.233333333,-0.25,-0.266666667,-0.283333333,-0.3,-0.316666667,-0.333333333,-0.35,-0.366666667,-0.383333333,-0.4,-0.416666667,-0.433333333,-0.45,-0.466666667,-0.483333333,-0.5,-0.48125,-0.4625,-0.44375,-0.425,-0.40625,-0.3875,-0.36875,-0.35,-0.33125,-0.3125,-0.29375,-0.275,-0.25625,-0.2375,-0.21875,-0.2,-0.18125,-0.1625,-0.14375,-0.125,-0.10625,-0.0875,-0.06875,-0.05,-0.03125,-0.0125,0.00625,0.025,0.04375,0.0625,0.08125,0.1,0.11875,0.1375,0.15625,0.175,0.19375,0.2125,0.23125,0.25,0.26875,0.2875,0.30625,0.325,0.34375,0.3625,0.38125,0.4,0.41875,0.4375,0.45625,0.475,0.49375,0.5125,0.53125,0.55,0.56875,0.5875,0.60625,0.625,0.64375,0.6625,0.68125,0.7,0.71875,0.7375,0.75625,0.775,0.79375,0.8125,0.83125,0.85,0.86875,0.8875,0.90625,0.925,0.94375,0.9625,0.98125,1,0.98125,0.9625,0.94375,0.925,0.90625,0.8875,0.86875,0.85,0.83125,0.8125,0.79375,0.775,0.75625,0.7375,0.71875,0.7,0.68125,0.6625,0.64375,0.625,0.60625,0.5875,0.56875,0.55,0.53125,0.5125,0.49375,0.475,0.45625,0.4375])

    depth_ex_signal_3 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

    depth_ex_signal_4 = np.array([-0.24220648, -0.24220648, -0.24220648, -0.21021166, -0.17821683,-0.21021166, -0.17821683, -0.17821683, -0.14622201, -0.17821683,-0.14622201, -0.14622201, -0.11422719, -0.11422719, -0.11422719,-0.17821683, -0.11422719, -0.11422719, -0.11422719, -0.08223237,-0.08223237, -0.05023755, -0.05023755, -0.05023755, -0.01824272,-0.01824272,  0.0137521 ,  0.0137521 ,  0.0137521 ,  0.04574692, 0.04574692,  0.04574692,  0.07774174,  0.07774174,  0.10973657, 0.14173139,  0.10973657,  0.14173139,  0.14173139,  0.17372621, 0.20572103,  0.17372621,  0.23771585,  0.23771585,  0.26971068, 0.3017055 ,  0.33370032,  0.39768997,  0.36569514,  0.39768997, 0.39768997,  0.42968479,  0.46167961,  0.46167961,  0.49367443, 0.49367443,  0.49367443,  0.55766408,  0.55766408,  0.5896589 , 0.62165372,  0.5896589 ,  0.62165372,  0.65364854,  0.65364854, 0.65364854,  0.71763819,  0.71763819,  0.74963301,  0.78162783, 0.78162783,  0.84561748,  0.8776123 ,  0.84561748,  0.94160194, 1.00559159,  1.10157605,  1.10157605,  1.1655657 ,  1.13357088, 1.13357088,  1.13357088,  1.1655657 ,  1.19756052,  1.1655657 , 1.13357088,  1.1655657 ,  1.13357088,  1.1655657 ,  1.1655657 , 1.13357088,  1.1655657 ,  1.13357088,  1.1655657 ,  1.13357088, 1.13357088,  1.13357088,  1.13357088,  1.13357088,  1.13357088, 1.1655657 ,  1.10157605,  1.13357088,  1.13357088,  1.13357088, 1.13357088,  1.13357088,  1.1655657 ,  1.10157605,  1.1655657 , 1.13357088,  1.13357088,  1.1655657 ,  1.13357088,  1.13357088, 1.13357088,  1.13357088,  1.13357088,  1.13357088,  1.13357088, 1.1655657 ,  1.10157605,  1.13357088,  1.1655657 ,  1.1655657 , 1.1655657 ,  1.13357088,  1.13357088,  1.10157605,  1.1655657 , 1.1655657 ,  1.13357088,  1.13357088,  1.1655657 ,  1.10157605, 1.10157605,  1.13357088,  1.13357088,  1.1655657 ,  1.10157605, 1.13357088,  1.13357088,  1.1655657 ,  1.10157605,  1.13357088, 1.13357088,  1.13357088,  1.13357088,  1.10157605,  1.13357088, 1.13357088,  1.1655657 ,  1.10157605,  1.13357088,  1.13357088, 1.10157605,  1.1655657 ,  1.10157605,  1.13357088,  1.13357088])
    depth_ex_signal_4 = (depth_ex_signal_4 - np.mean(depth_ex_signal_4)) / np.std(depth_ex_signal_4)
    
    vel_ex_signal_3 = np.array([ 0.35858966,  0.47811955,  0.47811955,  0.59764943,  0.47811955, 0.35858966,  0.71717932,  0.59764943,  0.47811955,  0.59764943, 0.59764943,  0.8367092 ,  0.8367092 ,  0.59764943,  0.71717932, 0.71717932,  0.8367092 ,  0.47811955,  0.59764943,  0.59764943, 0.71717932,  0.71717932,  0.59764943,  0.47811955,  0.47811955, 0.95623909,  0.8367092 ,  0.8367092 ,  0.95623909,  0.8367092 , 0.95623909,  0.8367092 ,  0.59764943,  0.71717932,  0.8367092 , 0.95623909,  0.71717932,  0.71717932,  0.95623909,  0.8367092 , 0.95623909,  0.8367092 ,  0.8367092 ,  1.07576898,  1.07576898, 1.19529886,  1.19529886,  1.19529886,  1.07576898,  1.43435864, 1.19529886,  1.31482875,  1.43435864,  1.55388852,  1.7929483 , 1.7929483 ,  1.7929483 ,  1.67341841,  1.7929483 ,  1.67341841, 1.55388852,  1.67341841,  1.67341841,  1.7929483 ,  1.67341841, 1.67341841,  1.55388852,  1.67341841,  1.55388852,  1.55388852, 1.43435864,  1.31482875,  1.31482875,  1.43435864,  1.31482875, 1.07576898,  0.95623909,  1.07576898,  1.19529886,  0.59764943, 0.8367092 ,  0.71717932,  0.71717932,  0.59764943,  0.35858966, 0.        , -0.11952989,  0.11952989,  0.        ,  0.        , 0.23905977, -0.11952989,  0.11952989, -0.23905977, -0.11952989,-0.11952989,  0.        , -0.11952989, -0.23905977, -0.11952989, 0.        , -0.11952989,  0.        , -0.23905977, -0.35858966,-0.11952989, -0.11952989, -0.11952989, -0.23905977,  0.        ,-0.11952989, -0.11952989, -0.11952989, -0.11952989, -0.11952989,-0.11952989,  0.        , -0.11952989, -0.11952989,  0.        ,-0.23905977,  0.11952989, -0.11952989,  0.        ,  0.        ,-0.11952989,  0.        , -0.11952989,  0.        , -0.11952989,-0.11952989,  0.        , -0.11952989, -0.11952989,  0.11952989,-0.11952989, -0.11952989, -0.11952989, -0.23905977, -0.11952989,-0.23905977, -0.11952989, -0.11952989,  0.        , -0.59764943,-0.47811955, -0.35858966, -0.35858966, -0.35858966, -0.35858966,-0.35858966, -0.59764943, -0.35858966, -0.35858966, -0.71717932,-0.59764943, -0.59764943, -0.59764943, -0.71717932, -0.59764943])
    vel_ex_signal_3 = (vel_ex_signal_3 - np.mean(vel_ex_signal_3)) / np.std(vel_ex_signal_3)


    # depth_ex_signal_2 = depth_ex_signal_2[::2]
    # vel_ex_signal_2 = vel_ex_signal_2[::2]


    convolve_features = np.zeros((len(depth), 3))
    for i in range(len(depth)):
        if(i >= int(convolve_window_size/2) and i < len(depth)-int(convolve_window_size/2)):
            # Normalize window
            d = depth[i-int(convolve_window_size/2):i+int(convolve_window_size/2)].copy()
            if(np.std(d) != 0):
                d = (d - np.mean(d)) / np.std(d)
            else:
                d = (d - np.mean(d))
            # d -= np.min(d)
            # if(np.max(d) != 0):
            #     d /= np.max(d)

            v = vel[i-int(convolve_window_size/2):i+int(convolve_window_size/2)]
            # if(np.std(v) != 0):
            #     v = (v - np.mean(v)) / np.std(v)
            # else:
            #     v = (v - np.mean(v))

            depth_c = np.correlate(d, depth_ex_signal_1[::ds])
            # vel_c = np.correlate(v, vel_ex_signal_2)
            
            convolve_features[i, 0] = np.max(depth_c) #np.sum(d * depth_c)#
            convolve_features[i, 1] = np.abs(np.var(v))# * np.var(v))#1/np.sum(np.abs(v - vel_ex_signal_3)) #np.max(vel_c)

            # if(convolve_features[i, 1] > 20000):
            #     convolve_features[i, 1] = 20000
            
            convolve_features[i, 2] = convolve_features[i, 1] * np.abs(np.var(d))

            convolve_features[i, 2] = -(convolve_features[i, 2] - 0.02) / 0.02
            if(convolve_features[i, 2] < 0):
                convolve_features[i, 2] = 0
                
            convolve_features[i, 1] /= 0.25

    # Normalize convolution features
    # convolve_features[:, 0] = (convolve_features[:, 0] - np.mean(convolve_features[:, 0])) / np.std(convolve_features[:, 0])
    # convolve_features[:, 1] = (convolve_features[:, 1] - np.mean(convolve_features[:, 1])) / np.std(convolve_features[:, 1])
    # convolve_features[:, 2] = (convolve_features[:, 2] - np.mean(convolve_features[:, 2])) / np.std(convolve_features[:, 2])

    # Minor data adjustments
    # convolve_features[:, 1] += 0.5
    # convolve_features[convolve_features[:, 1] < 0, 1] = 0

    """for i in range(len(depth)):
        if(i >= int(convolve_window_size/2) and i < len(depth)-int(convolve_window_size/2)):
            convolve_features[i, 2] = -convolve_features[i, 1] - 0.6#np.sum(convolve_features[i-int(convolve_window_size/2):i+int(convolve_window_size/2), 1])
            if(convolve_features[i, 2] < 0):
                convolve_features[i, 2] = 0"""
            

    # convolve_features[:, 0] *= convolve_features[:, 1]#-np.gradient(convolve_features[:, 0])
    # convolve_features[convolve_features[:, 1] < 0, 1] = 0

    # convolve_features[:, 0] = np.correlate(depth, depth_ex_signal_1[::2], mode='same')
    # convolve_features[:, 1] = np.correlate(acc, vel_ex_signal_2[::2], mode='same')

    return convolve_features