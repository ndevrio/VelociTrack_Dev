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

s_imu = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_cam1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_cam2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s_imu.settimeout(1.0)
s_cam1.settimeout(1.0)
s_cam2.settimeout(1.0)
s_imu.bind((HOST, PORT_IMU))
s_cam1.bind((HOST, PORT_CAM1))
s_cam2.bind((HOST, PORT_CAM2))


##########################################
####     Misc Function Definitions    ####
##########################################


plot_length = 1000#1500
ddata = np.zeros((plot_length,))
ddata_vel = np.zeros((plot_length,))
ddata_acc = np.zeros((plot_length,))
fdata = np.zeros((plot_length,))
odata = np.zeros((plot_length,))
flat_window = np.zeros((60,))


raw_acc_buffer = np.zeros((plot_length*10, 5)) # init with zero accel

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
    global space_pressed, current_class, training_instances, gt_active, KILL_THREADS, path_to_save, cam1_file, cam2_file, imu_file, trial_num, RECORDING
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
            trial_txt.setColor((255, 100, 100))
        else:
            cam1_file.close()
            cam2_file.close()
            imu_file.close()
            cam1_file = None
            cam2_file = None
            imu_file = None
            trial_txt.setColor((200, 200, 200))
            
            print("Saved: ", my_classes, training_instances)
            training_instances[:] = 0


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
dplot = win.addPlot(row=0, col=0, colspan=2, pen=(156, 39, 176))
dplot.setTitle("Finger Z depth", size="20pt")
dplot.setYRange(80, 250, padding=0)  # Updated y-axis range to 0-3000
depthPlotter = dplot.plot(pen=pg.mkPen(255, 100, 100, width=w))
dplot.enableAutoRange()

dplot = win.addPlot(row=1, col=0, colspan=2, pen=(156, 39, 176))
dplot.setTitle("Finger Z acc", size="20pt")
dplot.setYRange(-0.1, 1.1, padding=0)
depthPlotter2 = dplot.plot(pen=pg.mkPen(100, 255, 100, width=w))

dplot2 = win.addPlot(row=2, col=0, colspan=2, pen=(156, 39, 176))
dplot2.setTitle("IMU Z acc", size="20pt")
dplot2.setYRange(-10, 1000, padding=0)
#dplot2.setYRange(-500, 500, padding=0)
depthPlotter3 = dplot2.plot(pen=pg.mkPen(100, 255, 100, width=w))

class_vb = win.addViewBox(row=3, col=0)
class_vb.autoRange()
class_txt = pg.TextItem("No touch", anchor=(0.5,0.5))
class_txt.setColor((255, 255, 255))
class_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 20, QtGui.QFont.Bold))
class_txt.setPos(.5, .5)
class_vb.addItem(class_txt)
class_vb.setMaximumHeight(100)

trial_vb = win.addViewBox(row=3, col=1)
trial_vb.autoRange()
trial_txt = pg.TextItem("0", anchor=(0.5,0.5))
trial_txt.setColor((200, 200, 200))
trial_txt.setFont(QtGui.QFont("Bahnschrift SemiBold", 20, QtGui.QFont.Bold))
trial_txt.setPos(.5, .5)
trial_vb.addItem(trial_txt)
trial_vb.setMaximumHeight(100)


#####################################
####    Multithreading Section   ####
#####################################

KILL_THREADS = False


###############################################
####    Socket Read Function Definitions   ####
###############################################

source_counts = np.zeros((6,)).astype(int)

finger_x = np.zeros((2,))
finger_y = np.zeros((2,))
mp_did_reset = np.zeros((2,))
hand_tracked = np.zeros((2,))
def cam1_read():
    global KILL_THREADS, finger_x, finger_y, mp_did_reset, hand_tracked, cam1_file, RECORDING

    while not KILL_THREADS:
        try:
            data, _ = s_cam1.recvfrom(128)

            # write to cam1 file
            data = data.decode()

            if RECORDING:
                if cam1_file is not None:
                    cam1_file.write(data + "," + str(near_wall) + "\n")

            # print(data.split(","))
            finger_y[0] = float(data.split(",")[1])
            finger_x[0] = float(data.split(",")[2])
            mp_did_reset[0] = int(data.split(",")[3] == "True")
            hand_tracked[0] = int(data.split(",")[4] == "True")

            # only cam 2 processes the data
            # Process the data
            # process_data()
        except socket.timeout:
            pass

def cam2_read():
    global KILL_THREADS, finger_x, finger_y, mp_did_reset, hand_tracked, cam2_file, RECORDING

    while not KILL_THREADS:
        try:
            data, _ = s_cam2.recvfrom(128)
            data = data.decode()

            if RECORDING:
                if cam2_file is not None:
                    cam2_file.write(data + "," + str(near_wall) + "\n")

            finger_y[1] = float(data.split(",")[1])
            finger_x[1] = float(data.split(",")[2])
            mp_did_reset[1] = int(data.split(",")[3] == "True")
            hand_tracked[1] = int(data.split(",")[4] == "True")

            # Process the data
            process_data()
        except socket.timeout:
            pass

def imu_read():
    global raw_acc_buffer, KILL_THREADS, RECORDING, imu_file

    while not KILL_THREADS:
        acc = s_imu.recvfrom(1024)
        acc = pickle.loads(acc[0])
        raw_acc_buffer = np.roll(raw_acc_buffer, 1, axis=0)
        raw_acc_buffer[0] = acc

        if RECORDING:
            if imu_file is not None:
                for d in acc:
                    imu_file.write(f"{d},")
                imu_file.write(f"\n")
                
                
near_wall = 0
def unity_read():
    global KILL_THREADS, near_wall, odata
    
    host = ''  # Listen on all available interfaces
    port = 8080  # Choose a port number

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    while not KILL_THREADS:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
    
        while not KILL_THREADS:
            data = client_socket.recv(1024)
            if not data:
                break
            near_wall = int(data.decode())
            if(near_wall != 0):
                near_wall = 1
            response = f"Echo: {near_wall}"
            client_socket.send(response.encode())
            
            odata = np.roll(odata, 1)
            odata[0] = near_wall

        client_socket.close()
        print(f"Connection closed with {client_address}")


###################################################
####    Data Processing Function Definitions   ####
###################################################
start_time = 0
process_time = time.time()
last_val = 0
ddata_acc_old = 0
res_count = 0
one_euro_filter = OneEuroFilter(min_cutoff=1.5, beta=0.5)
def process_data():
    global process_time, finger_x, mp_did_reset, patch_diffs, ddata, fdata, odata, flat_window, near_wall, cam2_raw, last_val, source_counts, ddata_vel, ddata_acc, ddata_acc_old, res_count, one_euro_filter, start_time

    ### Get Z depth from stereo in images using camera geometry
    b = 71
    f_x = 100 #825

    #finger_x += 20

    if finger_x[1] - finger_x[0] == 0:
        return
    
    # fz = f_x * b / np.abs(finger_x[0] - finger_x[1])
    fz = finger_x[1] - finger_x[0]

    ddata = np.roll(ddata, 1)
    ddata[0] = fz
    
    ### Data filtering
    # if(fz > 10 and fz < 1000):
    #     if(start_time == 0):
    #         ddata[0] = one_euro_filter.filter(fz, 0)
    #         start_time = time.time()
    #     else:
    #         ddata[0] = one_euro_filter.filter(fz, time.time()-start_time)
    
    # thres = 5
    # if(np.abs(ddata[0] - ddata[1]) > thres and np.abs(ddata[2] - ddata[1]) > thres):
        
        
    #     ddata[1] = (ddata[2] + ddata[0]) / 2
    
    
    # Check did_reset value
    res = np.logical_or(mp_did_reset[0], mp_did_reset[1]).astype(np.int16)
    odata = np.roll(odata, 1)
    odata[0] = res

    ### Get velocity and acceleration (virtual IMU) data from Z depth
    # rval = 10
    rval = 15
    
    if(res == 1):
        res_count = rval*2
    else:
        res_count -= 1

    ddata_vel = ddata - np.roll(ddata, rval)
    ddata_vel = np.roll(ddata_vel, -rval)
    ddata_acc = ddata_vel - np.roll(ddata_vel, rval)
    ddata_acc = np.roll(ddata_vel, -rval)
    
    # ddata_acc = np.roll(ddata_acc, 1)
    # if(res_count >= 1):
    #     ddata_acc[0] = ddata_acc_old
    # else:    
    #     ddata_acc[0] = ddata_acc_new
    #     ddata_acc_old = ddata_acc_new
    
    # ddata_vel = np.diff(ddata, rval)
    # ddata_acc = np.diff(ddata_vel, rval)
    
    # ddata_filtered = np.convolve(ddata, np.ones(5)/5, mode='valid')
    ddata_filtered = np.convolve(ddata, np.ones(20)/20, mode='valid')

    # ddata_acc = np.diff(np.diff(ddata, rval), rval)[rval:-rval] / (1/800**2)
    # ddata_acc = np.diff(np.diff(ddata, rval), rval)[rval:-rval] / (800**2)
    # ddata_acc = np.diff(np.diff(ddata, rval), rval)[rval:-rval] / (800**2)
    # ddata_acc = np.diff(ddata, rval)
    # ddata_acc = np.diff(ddata_filtered, rval)
    # ddata_acc = ddata_filtered
    # ddata_acc = np.convolve(ddata_acc_raw, np.ones(15)/15, mode='valid')

##########################
####    Plot Update   ####
##########################
def update():
    global ddata, odata, raw_acc_buffer, reshape_time, ddata_vel, ddata_acc

    depthPlotter.setData(ddata)
    depthPlotter2.setData(ddata_vel) #odata
    #depthPlotter3.setData(odata)#np.linalg.norm(raw_acc_buffer[:, :-1], axis=1))
    depthPlotter3.setData(ddata_acc) #raw_acc_buffer[:, -1])
    
    ### Update trial counter
    if(RECORDING):
        training_instances[current_class] += 1
        trial_txt.setText(str(training_instances[current_class]))

        ### Data frame counting
        # if(time.time() - reshape_time > 1.0):
        #     reshape_to_window()

    ### Update ground truth text
    class_txt.setText(my_classes[current_class])


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.setInterval(33) # 1000ms / 33ms ≈ 30fps
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':

    t1 = threading.Thread(target=imu_read)
    t2 = threading.Thread(target=cam1_read)
    t3 = threading.Thread(target=cam2_read)
    t4 = threading.Thread(target=unity_read)
    t1.start()
    t2.start()
    t3.start()
    t4.start()

    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()
    finally:
        KILL_THREADS = True

        # Stop streaming
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        timer.stop()
        
        s_imu.close()
        s_cam1.close()
        s_cam2.close()