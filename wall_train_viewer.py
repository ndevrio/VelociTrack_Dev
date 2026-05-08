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
from funcs import combine_data

import torch
from mmapdataset import RealSenseDataset
from models_new import Depth1DCNN, TimeSeriesData


###############################
####    Data Load Section  ####
###############################
current_class = 0
my_classes = ['No touch', 'Touch']
training_instances = np.zeros((len(my_classes),), dtype=int)
space_pressed = False


infile = "./study_data/" + sys.argv[1] + "/" + sys.argv[2] + "_viz_data.npy"
data = np.load(infile, allow_pickle=True)

feature_window_size = 3000


############################
####    Model Section   ####
############################
# model = Depth1DCNN()
# model.load_state_dict(torch.load('./checkpoints/20250411-121855_p0_0_epoch=13-val_loss=0.08.ckpt')["state_dict"])
# model.zero_grad()
# model.eval()
# model.to('cuda')


##########################################
####     Misc Function Definitions    ####
##########################################

def forward():
    global cur_idx

    if(cur_idx < len(data)):
        cur_idx -= multiplier


def backward():
    global cur_idx

    if(cur_idx > 0):
        cur_idx += multiplier


class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


multiplier = 1
def keyPressed(evt):
    global space_pressed, cur_idx, training_instances, multiplier
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
        multiplier = 30
        mul_txt.setText("30x")
        mul_txt.setColor((100, 255, 100))
    elif(evt.key() == 52): # 4
        multiplier = 1000
        mul_txt.setText("1,000x")
        mul_txt.setColor((50, 255, 50))
    elif(evt.key() == 53): # 5
        multiplier = 10000
        mul_txt.setText("10,000x")
        mul_txt.setColor((0, 255, 0))
        

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
p1 = win.addPlot(row=0, col=0, colspan=2, title="", pen=(255, 153, 51))
p2 = win.addPlot(row=1, col=0, colspan=2, title="", pen=(253, 91, 120))
p3 = win.addPlot(row=2, col=0, colspan=2, title="", pen=(255, 96, 55))
p4 = win.addPlot(row=3, col=0, colspan=2, title="", pen=(255, 153, 102))
p5 = win.addPlot(row=4, col=0, colspan=2, title="", pen=(255, 153, 51))
p6 = win.addPlot(row=5, col=0, colspan=2, title="", pen=(255, 204, 51))
p7 = win.addPlot(row=6, col=0, colspan=2, title="", pen=(255, 255, 102))
p1.enableAutoRange(axis='y')
# p2.enableAutoRange(axis='y')
p2.setYRange(-.25, .25)
p3.setYRange(-.25, .25)
# p3.enableAutoRange(axis='y')
p4.enableAutoRange(axis='y')
p5.enableAutoRange(axis='y')
p6.setYRange(-.2, 1.2)
p7.setYRange(-.2, 1.2)
vline_pos=int(feature_window_size/2)
# p1.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p2.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p3.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p4.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p5.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p6.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p7.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
w = 2
c1 = p1.plot(pen=pg.mkPen(255, 255, 255, width=w))
c2 = p2.plot(pen=pg.mkPen(50, 255, 50, width=w))
c3 = p3.plot(pen=pg.mkPen(255, 50, 255, width=w))
c4 = p4.plot(pen=pg.mkPen(255, 50, 255, width=w))
c5 = p5.plot(pen=pg.mkPen(255, 50, 255, width=w))
c6 = p6.plot(pen=pg.mkPen(200, 200, 50, width=w))
c7 = p7.plot(pen=pg.mkPen(255, 255, 100, width=w))
p1.hideAxis('bottom')
p2.hideAxis('bottom')
p3.hideAxis('bottom')
p4.hideAxis('bottom')
p5.hideAxis('bottom')
p6.hideAxis('bottom')


class_vb = win.addViewBox(row=15, col=0, rowspan=2)
class_vb.autoRange()
class_txt = pg.TextItem("", anchor=(0.5,0.5))
class_txt.setColor((255, 255, 255))
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


#################################
####    Create predictions   ####
#################################

window_size = 320
predictions = np.zeros((len(data),))
batch_size = 128

c = 1

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

for i in tqdm(range(len(data)-500)):
    if(i >= int(window_size/2)):
        d = data[i-int(window_size/2):i+int(window_size/2), 5].copy()
        d = normalize_depth(d)
        depth_in[c % batch_size] = torch.from_numpy(d).float().unsqueeze(1).to('cuda')
        vel_in[c % batch_size] = torch.from_numpy(data[i-int(window_size/2):i+int(window_size/2), 6]).float().unsqueeze(1).to('cuda')
        
        conv_depth_in[c % batch_size] = torch.from_numpy(data[i-int(window_size/2):i+int(window_size/2), 8]).float().unsqueeze(1).to('cuda')
        conv_vel_in[c % batch_size] = torch.from_numpy(data[i-int(window_size/2):i+int(window_size/2), 9]).float().unsqueeze(1).to('cuda')
        conv_vel2_in[c % batch_size] = torch.from_numpy(data[i-int(window_size/2):i+int(window_size/2), 10]).float().unsqueeze(1).to('cuda')
        
        if((c % batch_size) == 0):
            # X = torch.stack((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in), dim=2)
            # # # X = torch.stack((depth_in, vel_in), dim=2)
            # X = X.flatten(start_dim=1)
            # result = model(X)[:, 0]
            
            result = model((depth_in, vel_in, conv_depth_in, conv_vel_in, conv_vel2_in))[:, 0]
            
            result = (torch.sigmoid(result) >= 0.5).cpu().numpy()
            # result = torch.argmax(result, 1).cpu().numpy()
            
            predictions[i-batch_size:i] = result.astype(int)
            # print(predictions[i])
        
        c += 1

predictions = np.load("predictions.npy")
# np.save("predictions.npy", predictions)


cur_idx = len(predictions) - 1

### Go through predictions and turn window into 1 if >80 of prediction is 1
mod_pred = np.zeros(predictions.shape)
window_size = 150
t = 0.025

state = 0
last_pred = 0
last_gt = 0
c_mat = np.zeros((2, 2))

for i in range(len(predictions)):
    if(i > int(window_size / 2) and i < (len(predictions) - int(window_size / 2) - 1) and data[i, 11] == 0):
        s = np.sum(predictions[i-int(window_size/2):i+int(window_size/2)])
        mod_pred[i] = int(s > t*window_size)

        pred = mod_pred[i]
        g = data[i, -1]
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
    elif(data[i, 11] == 1):
        state = 0

data = np.flip(data, axis=0)
mod_pred = np.flip(mod_pred, axis=0)
predictions = np.flip(predictions, axis=0)

##########################
####    Plot Update   ####
##########################
last_time = 0
frame_rate = 0.001
# last_cur_idx0 = -1
# counters = np.zeros((2,))
# count_limit = 3
# filt_class = 0
def update():
    global last_time, cur_idx, last_cur_idx0, counters, count_limit, filt_class, mod_pred

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

    if(plot_range_l < 0 or plot_range_r > len(data)):
        return
    
    # transformed_y = np.fft.fft(depth_data[plot_range_l:plot_range_r])
    # freqs_magnitude = np.abs(transformed_y)
    
    # depth = features[plot_range_l:plot_range_r, 1] - features[plot_range_l:plot_range_r, 5]
    # c2.setData(convolve_features[plot_range_l:plot_range_r, 0])
    # c2.setData(features[plot_range_l+d:plot_range_r+d, 1])
    # c3.setData(features[plot_range_l+d:plot_range_r+d, 5])
    c1.setData(data[plot_range_l:plot_range_r, 5])
    c2.setData(data[plot_range_l:plot_range_r, 6])
    c3.setData(data[plot_range_l:plot_range_r, 8])
    c4.setData(data[plot_range_l:plot_range_r, 9])
    c5.setData(data[plot_range_l:plot_range_r, 10])
    c6.setData(predictions[plot_range_l:plot_range_r])
    c7.setData(mod_pred[plot_range_l:plot_range_r])
    #mod_pred

    ### Update class
    #class_txt.setText(my_classes[int(classes[cur_idx])])

    ### Prediction
    # pred = int(model.predict(np.reshape(feature_data[cur_idx], (1, feature_data[cur_idx].size))))
    # class_txt.setText(my_classes[pred])


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QGuiApplication.instance().exec_()
