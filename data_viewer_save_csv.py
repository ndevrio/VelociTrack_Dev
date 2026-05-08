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


infile = "./study_data/" + sys.argv[1] + "/" + sys.argv[2] + "_viz_data.npy"
data = np.load(infile, allow_pickle=True)

feature_window_size = 500


depth = data[:, 4] - data[:, 2]
vel = depth - np.roll(depth, 15)
vel = np.roll(vel, -15)
acc = vel - np.roll(vel, 15)
acc = np.roll(acc, -15)


##########################################
####     Misc Function Definitions    ####
##########################################

def forward():
    global cur_idx

    if(cur_idx < len(data)):
        cur_idx += multiplier


def backward():
    global cur_idx

    if(cur_idx > 0):
        cur_idx -= multiplier


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
    elif(evt.key() == 50): # 2
        multiplier = 10
    elif(evt.key() == 51): # 3
        multiplier = 100
    elif(evt.key() == 52): # 4
        multiplier = 1000
    elif(evt.key() == 53): # 5
        multiplier = 10000
    elif(evt.key() == 83):
        save_csv()

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
p2 = win.addPlot(row=1, col=0, colspan=2, title="Velocity", pen=(253, 91, 120))
p3 = win.addPlot(row=2, col=0, colspan=2, title="Acceleration", pen=(255, 96, 55))
p1.enableAutoRange(axis='y')
# p1.setYRange(0, 100)
p2.enableAutoRange(axis='y')
p3.enableAutoRange(axis='y')
# vline_pos=int(feature_window_size/2)
# p1.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p2.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
# p3.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
c1 = p1.plot(pen=(255, 53, 94))
c2 = p2.plot(pen=(253, 91, 120))
c3 = p3.plot(pen=(255, 96, 55))
p1.hideAxis('bottom')
p2.hideAxis('bottom')


#######################
####    CSV Save   ####
#######################
def save_csv():
    global cur_idx

    plot_range_l = cur_idx-int(feature_window_size/2)
    plot_range_r = cur_idx+int(feature_window_size/2)

    if(plot_range_l < 0 or plot_range_r > len(data)):
        return

    depth_out = depth[plot_range_l:plot_range_r]
    vel_out = vel[plot_range_l:plot_range_r]
    acc_out = acc[plot_range_l:plot_range_r]

    data_out = np.stack((depth_out, vel_out, acc_out))
    np.savetxt("example_tap.csv", data_out, delimiter=",", fmt='%1.5f')



##########################
####    Plot Update   ####
##########################
last_time = 0
frame_rate = 0.001
def update():
    global last_time, cur_idx
    
    plot_range_l = cur_idx-int(feature_window_size/2)
    plot_range_r = cur_idx+int(feature_window_size/2)

    ### Update multiplier if in auto mode
    if(space_pressed):# and (time.time() - last_time) > frame_rate):
        forward()
        last_time = time.time()

    if(plot_range_l < 0 or plot_range_r > len(data)):
        return
    
    c1.setData(depth[plot_range_l:plot_range_r])
    c2.setData(vel[plot_range_l:plot_range_r])
    c3.setData(acc[plot_range_l:plot_range_r])


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
