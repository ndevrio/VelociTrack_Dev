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
import os

###############################
####    Data Load Section  ####
###############################
current_class = 0
my_classes = ['No touch', 'Touch']
training_instances = np.zeros((len(my_classes),), dtype=int)
space_pressed = False
cur_idx = 0


acc_file = "../study_data/" + sys.argv[1] + "/" + sys.argv[2] + "_stat_acc.bin"
label_file = "../study_data/" + sys.argv[1] + "/" + sys.argv[2] + "_stat_labels.bin"
gt = np.memmap(label_file, dtype='ubyte', mode='r')

ypred = np.load("p4_0_ypred.npy")
# gt = np.reshape(gt, (int(len(gt)/40), 40))

acc_shape = (len(gt), 40, 4)
data = np.memmap(acc_file, dtype='float32', mode='r+', shape=acc_shape)

###############################################
####     Calculate statistical features    ####
###############################################
from scipy.stats import skew, kurtosis

stat_features = np.zeros((len(data), 5))

# data = np.linalg.norm(data, axis=2)

# # [0] Maximum
# stat_features[:, 0] = np.max(data, axis=1)
# # [1] Minimum
# stat_features[:, 1] = np.min(data, axis=1)
# # [2] Mean
# stat_features[:, 2] = np.mean(data, axis=1)
# # [3] Minimum
# stat_features[:, 3] = skew(data, axis=1)
# # [4] Minimum
# stat_features[:, 4] = kurtosis(data, axis=1)

##########################################
####     Misc Function Definitions    ####
##########################################

def forward():
    global cur_idx

    if(cur_idx < len(data)):
        cur_idx += multiplier
        print(gt[cur_idx], ypred[cur_idx])


def backward():
    global cur_idx

    if(cur_idx > 0):
        cur_idx -= multiplier
        print(gt[cur_idx], ypred[cur_idx])


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


#########################
####    Qt Section   ####
#########################

app = QtWidgets.QApplication

### Data dashboard window
win = KeyPressWindow(show=True, title="Basic plotting examples", border=(50,50,50))
win.sigKeyPress.connect(keyPressed)
win.resize(1300, 900)
win.setWindowTitle('RealSense Data Player')
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

### Add EMG plots
p1 = win.addPlot(row=0, col=0, colspan=2, title="Depth", pen=(255, 153, 51))
p2 = win.addPlot(row=1, col=0, colspan=2, title="Max", pen=(253, 91, 120))
p3 = win.addPlot(row=2, col=0, colspan=2, title="Min", pen=(255, 96, 55))
p4 = win.addPlot(row=3, col=0, colspan=2, title="Mean", pen=(255, 153, 102))
p5 = win.addPlot(row=4, col=0, colspan=2, title="Skew", pen=(255, 204, 51))
p6 = win.addPlot(row=5, col=0, colspan=2, title="Kurtosis", pen=(255, 255, 102))
p7 = win.addPlot(row=6, col=0, colspan=2, title="Ground truth", pen=(255, 255, 102))
# p1.enableAutoRange(axis='y')
p1.setYRange(-2, 2)
p2.setYRange(-2, 2)
p3.setYRange(-2, 2)
# p4.setYRange(200, 600)
p4.setYRange(0, 5)
p7.setYRange(-.2, 1.2)
vline_pos=int(40/2)
p1.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p2.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p3.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p4.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p5.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p7.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p6.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
c1 = p1.plot(pen=(255, 53, 94))
c2 = p2.plot(pen=(253, 91, 120))
c3 = p3.plot(pen=(255, 96, 55))
c4 = p4.plot(pen=(255, 153, 102))
c5 = p5.plot(pen=(255, 153, 51))
c6 = p6.plot(pen=(255, 255, 255))
c7 = p7.plot(pen=(50, 255, 50)) #(255, 255, 102)
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


##########################
####    Plot Update   ####
##########################
last_time = 0
frame_rate = 0.001
def update():
    global last_time, cur_idx
    
    plot_range_l = cur_idx-int(40/2)
    plot_range_r = cur_idx+int(40/2)

    ### Update multiplier if in auto mode
    if(space_pressed):# and (time.time() - last_time) > frame_rate):
        forward()
        last_time = time.time()

    if(plot_range_l < 0 or plot_range_r > len(data)):
        return
    
    x = np.linalg.norm(data[:, :, :-1], axis=2)
    
    c1.setData(data[cur_idx, :, 0])
    c2.setData(data[cur_idx, :, 1])
    c3.setData(data[cur_idx, :, 2])
    # c4.setData(data[cur_idx, :, 3])
    c4.setData(x[cur_idx])
    # c3.setData(stat_features[plot_range_l:plot_range_r, 3])
    # c4.setData(stat_features[plot_range_l:plot_range_r, 6])
    # c5.setData(stat_features[plot_range_l:plot_range_r, 9])
    # c6.setData(stat_features[plot_range_l:plot_range_r, 12])
    # # c7.setData(data[plot_range_l:plot_range_r, -1])
    # # c7.setData(valid[plot_range_l:plot_range_r])
    # c7.setData(gt[cur_idx, :])
    
    
    acc_win = data[:, -5:-2]
    # valid = data[:, 11]
    # gt = data[:, -1]


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
