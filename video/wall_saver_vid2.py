import sys
import numpy as np
import time
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import numpy as np
import pyqtgraph as pg
import socket
import pickle
import threading
import serial
from pathlib import Path
import math


###############################
####    Data Save Section  ####
###############################
current_class = 0
my_classes = ['No touch', 'Touch']
training_instances = np.zeros((len(my_classes),), dtype=int)
space_pressed = False
gt_active = True

cam2_raw = []
cam2_data = []
acc_data = []
RECORDING = False
source_counts = np.zeros((2,)).astype(int)

########################################
####    Socket Connection Section   ####
########################################

import socket

HOST = "127.0.0.1"
PORT_CAM1 = 8081
PORT_CAM2 = 8082
PORT_IMU = 8083

# s_imu = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_cam1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_cam2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# s_imu.settimeout(1.0)
s_cam1.settimeout(1.0)
s_cam2.settimeout(1.0)
# s_imu.bind((HOST, PORT_IMU))
s_cam1.bind((HOST, PORT_CAM1))
s_cam2.bind((HOST, PORT_CAM2))


##########################################
####     Misc Function Definitions    ####
##########################################


plot_length = 2000#1500
cam_data_dep = np.zeros((plot_length,))
cam_data_vel = np.zeros((plot_length,))
cam_data_acc = np.zeros((plot_length,))

imu_data_dep = np.zeros((plot_length,))
imu_data_vel = np.zeros((plot_length,))
imu_data_acc = np.zeros((plot_length,))


cam1_file = None
cam2_file = None
imu_file = None

class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

trial_num = 0

def keyPressed(evt):
    global space_pressed, current_class, training_instances, gt_active, KILL_THREADS, path_to_save, cam1_file, cam2_file, imu_file, trial_num, RECORDING, imu_data_vel, imu_data_dep
    if(evt.key() == QtCore.Qt.Key_Q): # 'q' event
        print('quit')
        KILL_THREADS = True
        QtWidgets.QApplication.instance().quit()
        sys.exit(0)
    elif(evt.key() == QtCore.Qt.Key_Space): # 'space' event
        RECORDING = not RECORDING
        if RECORDING:
            trial_num += 1
            cam1_file = open(path_to_save / f"cam1_data_{trial_num}.txt", "w")
            cam2_file = open(path_to_save / f"cam2_data_{trial_num}.txt", "w")
            imu_file = open(path_to_save / f"imu_data_{trial_num}.txt", "w")
        else:
            cam1_file.close()
            cam2_file.close()
            imu_file.close()
            cam1_file = None
            cam2_file = None
            imu_file = None
            
            print("Saved: ", my_classes, training_instances)
            training_instances[:] = 0
    elif(evt.key() == QtCore.Qt.Key_R): # 'r' event
        imu_data_vel[:] = 0
        imu_data_dep[:] = 0



# get a name for the participant
participant_name = "g"#input("Enter participant name: ")
path_to_save = Path(f"./pilot_data/{participant_name}_{time.strftime('%Y%m%d_%H%M%S')}")
path_to_save.mkdir(parents=True, exist_ok=True)

#########################
####    Qt Section   ####
#########################

app = QtWidgets.QApplication

### Data dashboard window
win = KeyPressWindow(show=True, title="Basic plotting examples", border=(50,50,50))
win.sigKeyPress.connect(keyPressed)
win.resize(1280, 960)
win.setWindowTitle('RealSense Data Player')
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

## Add data plots
w = 1
cam_plot1 = win.addPlot(row=0, col=0, colspan=1, pen=(156, 39, 176))
# cam_plot1.setTitle("Camera depth", size="20pt")
cam_plot1.setLabel('left', "Depth")
# cam_plot1.setYRange(-100, 100, padding=0)  # Updated y-axis range to 0-3000

cam_plot2 = win.addPlot(row=1, col=0, colspan=1, pen=(156, 39, 176))
# cam_plot2.setTitle("Camera velocity", size="20pt")
cam_plot2.setLabel('left', "Velocity")
cam_plot2.setYRange(-20, 20, padding=0)


font=QtGui.QFont()
font.setPixelSize(65)
cam_plot1.getAxis("left").label.setFont(font)
cam_plot2.getAxis("left").label.setFont(font)


camPlotter1 = cam_plot1.plot(pen=pg.mkPen(255, 255, 255, width=w))
cam_plot1.enableAutoRange()
camPlotter2 = cam_plot2.plot(pen=pg.mkPen(100, 255, 100, width=w))
# cam_plot2.enableAutoRange()

#####################################
####    Multithreading Section   ####
#####################################

KILL_THREADS = False


###############################################
####    Socket Read Function Definitions   ####
###############################################

finger_x = np.zeros((2,))
finger_y = np.zeros((2,))
raw_acc_buffer = np.zeros((5,))
def cam1_read():
    global KILL_THREADS, finger_x, finger_y, cam1_file, RECORDING

    while not KILL_THREADS:
        try:
            data, _ = s_cam1.recvfrom(128)

            # write to cam1 file
            data = data.decode()

            if RECORDING:
                if cam1_file is not None:
                    cam1_file.write(data + "\n")

            # print(data.split(","))
            finger_y[0] = float(data.split(",")[1])
            finger_x[0] = float(data.split(",")[2])

        except socket.timeout:
            pass

def cam2_read():
    global KILL_THREADS, finger_x, finger_y, cam2_file, RECORDING

    while not KILL_THREADS:
        try:
            data, _ = s_cam2.recvfrom(128)
            data = data.decode()

            if RECORDING:
                if cam2_file is not None:
                    cam2_file.write(data + "\n")

            finger_y[1] = float(data.split(",")[1])
            finger_x[1] = float(data.split(",")[2])

            # Process the data
            process_data()
        except socket.timeout:
            pass

def imu_read():
    global raw_acc_buffer, KILL_THREADS, RECORDING, imu_file

    while not KILL_THREADS:
        acc = s_imu.recvfrom(1024)
        acc = pickle.loads(acc[0])

        # Read in acc data from IMU
        raw_acc_buffer = acc

        if RECORDING:
            if imu_file is not None:
                for d in acc:
                    imu_file.write(f"{d},")
                imu_file.write(f"\n")


###################################################
####    Data Processing Function Definitions   ####
###################################################
def process_data():
    global finger_x, raw_acc_buffer, cam_data_dep, cam_data_vel, cam_data_acc, imu_data_acc, imu_data_vel, imu_data_dep

    if finger_x[1] - finger_x[0] == 0:
        return
    
    fz = finger_x[1] - finger_x[0]

    cam_data_dep = np.roll(cam_data_dep, 1)
    cam_data_dep[0] = fz

    ### Get velocity and acceleration (virtual IMU) data from Z depth
    rval = 15

    cam_data_vel = -cam_data_dep + np.roll(cam_data_dep, rval)
    cam_data_vel = np.roll(cam_data_vel, -rval)
    cam_data_acc = cam_data_vel - np.roll(cam_data_vel, rval)
    cam_data_acc = np.roll(cam_data_acc, -rval)


    ### Get values from IMU data
    imu_data_acc = np.roll(imu_data_acc, 1)
    imu_data_acc[0] = raw_acc_buffer[3]

    # if(np.abs(imu_data_acc[0]) < 0.3):
        # imu_data_acc[0] = 0

    dt = 1/790
    imu_data_vel = np.roll(imu_data_vel, 1)
    imu_data_vel[0] = imu_data_vel[1] + imu_data_acc[0] * dt

    # if(np.abs(imu_data_vel[0]) < 0.):
        # imu_data_vel[0] = 0

    imu_data_dep = np.roll(imu_data_dep, 1)
    imu_data_dep[0] = imu_data_dep[1] + imu_data_vel[0] * dt



##########################
####    Plot Update   ####
##########################
def update():
    camPlotter1.setData(cam_data_dep)
    camPlotter2.setData(cam_data_vel)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.setInterval(33) # 1000ms / 33ms ≈ 30fps
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':

    # t1 = threading.Thread(target=imu_read)
    t2 = threading.Thread(target=cam1_read)
    t3 = threading.Thread(target=cam2_read)
    # t4 = threading.Thread(target=unity_read)
    # t1.start()
    t2.start()
    t3.start()
    # t4.start()

    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()
    finally:
        KILL_THREADS = True

        # Stop streaming
        # t1.join()
        t2.join()
        t3.join()
        # t4.join()
        timer.stop()
        
        # s_imu.close()
        s_cam1.close()
        s_cam2.close()