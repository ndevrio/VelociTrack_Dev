import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from matplotlib import cm
from sklearn.ensemble import RandomForestClassifier
import sys

###############################
####    Data Load Section  ####
###############################
current_class = 0
my_classes = ['No touch', 'Touch']
training_instances = np.zeros((len(my_classes),), dtype=int)
space_pressed = False
cur_idx = 0
has_imu = True

### Data descriptions:
### [Timestamp] [Finger X] [Finger Y] [Mediapipe reset] [Unity near wall]
### [Finger X] [Finger Y] [Mediapipe reset] [Unity near wall]
### [IMU X] [IMU Y] [IMU Z] [Pressure]
"""cam1_df = pd.read_csv(".\\study_data\\" + sys.argv[1] + "\\cam1_data_" + sys.argv[2] + ".txt", sep=',', header=None)
cam2_df = pd.read_csv(".\\study_data\\" + sys.argv[1] + "\\cam2_data_" + sys.argv[2] + ".txt", sep=',', header=None)

# Merge three files into one
features = pd.merge_asof(cam1_df, cam2_df, 0, direction="nearest")
features = features.astype(float).to_numpy()"""
cam1_df = pd.read_csv(".\\study_data\\" + sys.argv[1] + "\\cam1_data_" + sys.argv[2] + ".txt", sep=',', header=None)
cam2_df = pd.read_csv(".\\study_data\\" + sys.argv[1] + "\\cam2_data_" + sys.argv[2] + ".txt", sep=',', header=None)
try:
    imu_df = pd.read_csv(".\\study_data\\" + sys.argv[1] + "\\imu_data_" + sys.argv[2] + ".txt", sep=',', header=None)
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


mp_did_reset = np.logical_or(features[:, 3], features[:, 8]).astype(np.int16)
mp_hand_tracked = np.logical_and(features[:, 4], features[:, 9]).astype(np.int16)
near_wall = features[:, 5]

gt = np.zeros((features.shape[0],))#np.load(".\\study_data\\" + sys.argv[1] + "\\gt_data_" + sys.argv[2] + ".npy") #np.zeros((features.shape[0],))
valid = np.zeros((features.shape[0],))#np.load(".\\study_data\\" + sys.argv[1] + "\\valid_data_" + sys.argv[2] + ".npy") #np.zeros((features.shape[0],))

###########################
###   Feature creation  ###
###########################

def featurize():
    global features, near_wall
    
    ### [1] Shift amount for time difference between IMU and camera data
    d = 50 
    features[:, -4:] = np.roll(features[:, -4:], d, axis=0)
    

    ### [2] Convert pressure sensor data to binary

    gt_old = np.zeros((features.shape[0],))#features[:, -2].copy()
    # gt_old[gt_old < 265] = 0
    # gt_old[gt_old > 0] = 1
    
    d = 50
    near_wall = np.roll(near_wall, -d, axis=0)
    

    ### [3] Convert Y values from each camera into a single depth value using camera geometry
    diff = features[:, 7].copy() - features[:, 2].copy()

    # Correct for any divide by zero error
    for i in range(len(diff)):
        if diff[i] == 0:
            diff[i] = diff[i-1]

    # Account for flipped values in first three participants
    if(sys.argv[1] == "p1" or (sys.argv[1] == "p2" and int(sys.argv[2]) != 3 and int(sys.argv[2]) != 4) or sys.argv[1] == "p3"):
        depth = -diff
    else:
        depth = diff


    ### [4] Filter the depth data using a OneEuroFilter
    import math
    class OneEuroFilter:
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
    # for i in range(len(features)):
    #     # flat_window = np.roll(flat_window, 1)
    #     # flat_window[0] = depth[i]
        
    #     # [A] OneEuroFilter data filtering
    #     if not np.isinf(depth[i]):
    #         if(i == 0):
    #             depth[i] = one_euro_filter.filter(depth[i], 0)
    #         else:
    #             depth[i] = one_euro_filter.filter(depth[i], features[i, 0]-start_time)
        
    #     # [B] Custom features
    #     # if(i >= window_size): # If the window is valid
    #         ### [1] Finger is not shaking for X s within the window
    #         ### STD of window is < Z
    #         # new_features[i, 0] = int(np.abs(flat_window[-1] - flat_window[0]) < 3.0)
    #         # if(np.abs(flat_window[-1] - flat_window[0]) < 3.0):
    #         #     new_features[i, 0] = 1
    #         # else:
    #         #     new_features[i, 0] = 0
    #         # new_features[i, 0] = int(np.median(flat_window) == 0)
            

    ### [5] Derive velocity and acceleration data from depth
    r_val = 30
    velocity = depth - np.roll(depth, r_val)
    acc = velocity - np.roll(velocity, r_val)
    
    
    return depth, velocity, acc, gt_old


depth, velocity, acc, gt_old = featurize()

# gt = gt_old.copy()
# gt[gt < 310] = 0
# gt[gt > 0] = 1

"""### [1] Finger is near a surface
### Look to see if the absolute diff of depth between the centroid and average depth
### in a local patch around the finger is < X
patch_diff = pd
f1 = patch_diff#int(patch_diff < 200)


### [2] Finger has moved to start a tap
### Look to see if the STD of a recent window is > Y for ANY of the last N frames
move_thres = 100
if(np.std(flat_window) > move_thres):
    fdata = np.roll(fdata, 1)
    fdata[0] = 2
else:
    fdata = np.roll(fdata, 1)
    fdata[0] = 0

f2 = int(np.any(fdata[:25] == 2))


### [3] Finger is not shaking for X s within the window
### STD of window is < Z
thres = 40

f3 = int(np.std(flat_window) < thres)


### [4] Sharp spike is detected in the gradient plot when touching a surface
### or rebounding in the air.
grad_data = np.gradient(ddata) #ddata - np.roll(ddata, 1) #

f4 = int((grad_data[0] > 150) or (grad_data[0] < -150))"""


gt_set_val = 0
gt_set_enabled = False
bad_set_enabled = False
feature_window_size = 6000

##########################################
####     Misc Function Definitions    ####
##########################################

def forward():
    global cur_idx, gt, gt_set_enabled, gt_set_val, valid, bad_set_enabled, near_wall

    if(cur_idx < len(features)):
        if(gt_set_enabled):
            near_wall[cur_idx:cur_idx+multiplier] = gt_set_val
        # if(bad_set_enabled):
        #     valid[cur_idx:cur_idx+multiplier] = gt_set_val
        cur_idx += multiplier
    
    print(cur_idx)


def backward():
    global cur_idx, gt, gt_set_enabled, gt_set_val, valid, bad_set_enabled

    if(cur_idx > 0):
        if(gt_set_enabled):
            near_wall[cur_idx-multiplier:cur_idx] = gt_set_val
        # if(bad_set_enabled):
        #     valid[cur_idx-multiplier:cur_idx] = gt_set_val
        cur_idx -= multiplier
    
    print(cur_idx)
    
    
def save_gt():
    global gt, valid, near_wall
    
    np.save(".\\study_data\\" + sys.argv[1] + "\\gt_data_" + sys.argv[2] + ".npy", gt)
    np.save(".\\study_data\\" + sys.argv[1] + "\\valid_data_" + sys.argv[2] + ".npy", valid)
    np.save(".\\study_data\\" + sys.argv[1] + "\\wall_data_" + sys.argv[2] + ".npy", near_wall)
    print("Saved data: ", gt.shape, valid.shape)


class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


multiplier = 1
def keyPressed(evt):
    global space_pressed, cur_idx, training_instances, multiplier, gt_set_val, gt_set_enabled, bad_set_enabled
    if(evt.key() == 32): # space bar event
        space_pressed = ~space_pressed
        """if(space_pressed):
            mul_txt.setColor((255, 100, 100))
        else:
            mul_txt.setColor((200, 200, 200))"""
        if(space_pressed):
            mode_txt.setText("AUTO")
            mode_txt.setColor((50, 255, 150))
        else:
            mode_txt.setText("MANUAL")
            mode_txt.setColor((255, 150, 50))
    elif(evt.key() == 16777234): # left arrow
        backward()
        #class_txt.setText(my_classes[current_class])
        #mul_txt.setText(str(training_instances[current_class]))
    elif(evt.key() == 16777236): # right arrow
        forward()
        #class_txt.setText(my_classes[current_class])
        #mul_txt.setText(str(training_instances[current_class]))
    elif(evt.key() == 49): # 1
        multiplier = 1
        mul_txt.setText("1x")
        mul_txt.setColor((200, 255, 200))
    elif(evt.key() == 50): # 2
        multiplier = 10
        mul_txt.setText("10x")
        mul_txt.setColor((150, 255, 150))
    elif(evt.key() == 51): # 3
        multiplier = 100
        mul_txt.setText("100x")
        mul_txt.setColor((100, 255, 100))
    elif(evt.key() == 52): # 4
        multiplier = 1000
        mul_txt.setText("1,000x")
        mul_txt.setColor((50, 255, 50))
    elif(evt.key() == 53): # 5
        multiplier = 10000
        mul_txt.setText("10,000x")
        mul_txt.setColor((0, 255, 0))
    elif(evt.key() == 69): # e
        bad_set_enabled = 0
        gt_set_enabled = not gt_set_enabled
        if(gt_set_enabled):
            class_txt.setColor((50, 255, 50))
        else:
            class_txt.setColor((50, 50, 50))
    elif(evt.key() == 87): # w
        class_txt.setText("1")
        gt_set_val = 1
    elif(evt.key() == 83): # s
        class_txt.setText("0")
        gt_set_val = 0
    elif(evt.key() == 82): # r
        gt_set_enabled = 0
        bad_set_enabled = not bad_set_enabled
        if(bad_set_enabled):
            class_txt.setColor((50, 255, 255))
        else:
            class_txt.setColor((50, 50, 50))
    elif(evt.key() == 81): # q
        save_gt()

##########################
####    Model Setup   ####
##########################
        
# npzfile_train = np.load(sys.argv[1], allow_pickle=True)
# features_train = np.array(npzfile_train['cam_data'])
# features_train = np.reshape(features_train, (features_train.shape[0]*features_train.shape[1], features_train.shape[2]))

# # Create structured data
# y = features_train[:, 1].astype(int)
# X = features_train[:, 3:]

# model = RandomForestClassifier()

# model.fit(X, y)

# pred_data = model.predict(feature_data)

# ### Prediction data smoothing
# pred_data_raw = pred_data.copy()
# pred_data = np.zeros(pred_data.shape)
# smooth_size = 50
# ratio = 0.4
# for i in range(smooth_size, len(pred_data)):
#     pred_data[i] = int(np.sum(pred_data_raw[i-smooth_size:i]) > (smooth_size*ratio))

#########################
####    Qt Section   ####
#########################

app = QtWidgets.QApplication

### Data dashboard window
win = KeyPressWindow(show=True, title="Basic plotting examples", border=(50,50,50))
win.sigKeyPress.connect(keyPressed)
win.resize(1300, 800)
win.setWindowTitle('RealSense Data Player')
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

### Add EMG plots
p1 = win.addPlot(row=0, col=0, colspan=2, title="Raw depth data", pen=(255, 153, 51))
p2 = win.addPlot(row=1, col=0, colspan=2, title="mp_did_reset", pen=(253, 91, 120))
p3 = win.addPlot(row=2, col=0, colspan=2, title="mp_hand_tracked", pen=(255, 96, 55))
p4 = win.addPlot(row=3, col=0, colspan=2, title="near_wall", pen=(255, 153, 102))
p5 = win.addPlot(row=4, col=0, colspan=2, title="Pressure sensor", pen=(255, 153, 51))
p6 = win.addPlot(row=5, col=0, colspan=2, title="GT", pen=(255, 204, 51))
p7 = win.addPlot(row=6, col=0, colspan=2, title="Bad region", pen=(255, 255, 102))
p1.enableAutoRange(axis='y')
# p1.setYRange(0, 100)
p2.enableAutoRange(axis='y')
p3.enableAutoRange(axis='y')
p4.enableAutoRange(axis='y')
p5.enableAutoRange(axis='y')
p4.setYRange(-.2, 1.2)
# p5.setYRange(-.2, 1.2)
p6.setYRange(-.2, 1.2)
p7.setYRange(-.2, 1.2)
vline_pos=int(feature_window_size/2)
p1.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p2.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p3.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p4.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p5.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p6.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p7.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
c1 = p1.plot(pen=(255, 53, 94))
c2 = p2.plot(pen=(253, 91, 120))
c3 = p3.plot(pen=(255, 96, 55))
c4 = p4.plot(pen=(255, 153, 102))
c5 = p5.plot(pen=(255, 153, 51))
c6 = p6.plot(pen=(50, 255, 50)) #(255, 204, 51)
c7 = p7.plot(pen=(50, 255, 255)) #(255, 255, 102)
p1.hideAxis('bottom')
p2.hideAxis('bottom')
p3.hideAxis('bottom')
p4.hideAxis('bottom')
p5.hideAxis('bottom')
p6.hideAxis('bottom')


class_vb = win.addViewBox(row=15, col=0, rowspan=2)
class_vb.autoRange()
class_txt = pg.TextItem("", anchor=(0.5,0.5))
class_txt.setColor((50, 50, 50))
class_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 20, QtGui.QFont.Bold))
class_txt.setPos(.5, .5)
class_vb.addItem(class_txt)

mode_vb = win.addViewBox(row=15, col=1)
mode_vb.autoRange()
mode_txt = pg.TextItem("MANUAL", anchor=(0.5,0.5))
mode_txt.setColor((255, 100, 50))
mode_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 15, QtGui.QFont.Bold))
mode_txt.setPos(.5, .5)
mode_vb.addItem(mode_txt)
mode_vb.setMaximumWidth(225)

mul_vb = win.addViewBox(row=16, col=1)
mul_vb.autoRange()
mul_txt = pg.TextItem("1x", anchor=(0.5,0.5))
mul_txt.setColor((200, 255, 200))
mul_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 20, QtGui.QFont.Bold))
mul_txt.setPos(.5, .5)
mul_vb.addItem(mul_txt)
mul_vb.setMaximumWidth(225)


##########################
####    Plot Update   ####
##########################
last_time = 0
frame_rate = 0.001
def update():
    global last_time, cur_idx, gt, gt_old, gt_set_val, mp_did_reset, mp_hand_tracked, near_wall

    try:
        # print(cur_idx)

        ### EMG plots
        # plot_range_l = cur_idx-100#-int(feature_window_size/2)
        # plot_range_r = cur_idx#+int(feature_window_size/2)
        
        plot_range_l = cur_idx-int(feature_window_size/2)
        plot_range_r = cur_idx+int(feature_window_size/2)

        ### Update multiplier if in auto mode
        if(space_pressed):# and (time.time() - last_time) > frame_rate):
            forward()
            last_time = time.time()

        if(plot_range_l < 0 or plot_range_r > len(features)):
            return
        
        # transformed_y = np.fft.fft(depth_data[plot_range_l:plot_range_r])
        # freqs_magnitude = np.abs(transformed_y)
        
        # depth = features[plot_range_l:plot_range_r, 1] - features[plot_range_l:plot_range_r, 5]
        c1.setData(depth[plot_range_l:plot_range_r])
        # c2.setData(features[plot_range_l+d:plot_range_r+d, 1])
        # c3.setData(features[plot_range_l+d:plot_range_r+d, 5])
        c2.setData(mp_did_reset[plot_range_l:plot_range_r])
        c3.setData(mp_hand_tracked[plot_range_l:plot_range_r])
        
        # imu_acc = np.sqrt(np.exp2(features[plot_range_l:plot_range_r, -5]) + np.exp2(features[plot_range_l:plot_range_r, -4]) + np.exp2(features[plot_range_l:plot_range_r, -3]))
        
        c4.setData(near_wall[plot_range_l:plot_range_r])
        c5.setData(gt_old[plot_range_l:plot_range_r])
        c6.setData(gt[plot_range_l:plot_range_r])
        c7.setData(valid[plot_range_l:plot_range_r])

        ### Update class
        #class_txt.setText(my_classes[int(classes[cur_idx])])

        ### Prediction
        # pred = int(model.predict(np.reshape(feature_data[cur_idx], (1, feature_data[cur_idx].size))))
        # class_txt.setText(my_classes[pred])
    except IndexError:
        pass


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
