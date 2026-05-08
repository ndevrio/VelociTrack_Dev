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

feature_window_size = 2500

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
p2 = win.addPlot(row=1, col=0, colspan=2, title="Velocity", pen=(253, 91, 120))
p3 = win.addPlot(row=2, col=0, colspan=2, title="Acceleration", pen=(255, 96, 55))
p4 = win.addPlot(row=3, col=0, colspan=2, title="Depth conv", pen=(255, 153, 102))
p5 = win.addPlot(row=4, col=0, colspan=2, title="Acc conv", pen=(255, 153, 51))
p6 = win.addPlot(row=5, col=0, colspan=2, title="Invalid", pen=(255, 204, 51))
p7 = win.addPlot(row=6, col=0, colspan=2, title="Near wall", pen=(255, 255, 102))
p8 = win.addPlot(row=7, col=0, colspan=2, title="Ground truth", pen=(255, 255, 102))
p1.enableAutoRange(axis='y')
# p1.setYRange(0, 100)
p2.enableAutoRange(axis='y')
p3.enableAutoRange(axis='y')
p4.enableAutoRange(axis='y')
p5.enableAutoRange(axis='y')
p4.setYRange(-100, 100)
p5.setYRange(-0, .1)
p6.setYRange(-.2, 1.2)
p7.setYRange(-.2, 1.2)
p8.setYRange(-.2, 1.2)
vline_pos=int(feature_window_size/2)
p1.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p2.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p3.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p4.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p5.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p6.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p7.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
p8.addItem(pg.InfiniteLine(pos=vline_pos, pen=(255,255,255)))
c1 = p1.plot(pen=(255, 53, 94))
c2 = p2.plot(pen=(253, 91, 120))
c3 = p3.plot(pen=(255, 96, 55))
c4 = p4.plot(pen=(255, 153, 102))
c5 = p5.plot(pen=(255, 153, 51))
c6 = p6.plot(pen=(255, 255, 255)) #(255, 204, 51)
c7 = p7.plot(pen=(50, 255, 255)) #(255, 255, 102)
c8 = p8.plot(pen=(50, 255, 50)) #(255, 255, 102)
p1.hideAxis('bottom')
p2.hideAxis('bottom')
p3.hideAxis('bottom')
p4.hideAxis('bottom')
p5.hideAxis('bottom')
p6.hideAxis('bottom')
p7.hideAxis('bottom')


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
    
    plot_range_l = cur_idx-int(feature_window_size/2)
    plot_range_r = cur_idx+int(feature_window_size/2)

    ### Update multiplier if in auto mode
    if(space_pressed):# and (time.time() - last_time) > frame_rate):
        forward()
        last_time = time.time()

    if(plot_range_l < 0 or plot_range_r > len(data)):
        return
    
    c1.setData(data[plot_range_l:plot_range_r, 5])
    c2.setData(data[plot_range_l:plot_range_r, 6])
    c3.setData(data[plot_range_l:plot_range_r, 7])
    c4.setData(data[plot_range_l:plot_range_r, 11])
    c5.setData(data[plot_range_l:plot_range_r, 12])
    c6.setData(data[plot_range_l:plot_range_r, 13])
    c7.setData(data[plot_range_l:plot_range_r, 14])
    c8.setData(data[plot_range_l:plot_range_r, -1])


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()
