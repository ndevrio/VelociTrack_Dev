import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

ser = serial.Serial('COM18', 1000000)

samp_rate = 60
num_win_samples = 2000
window_size = int(num_win_samples * (samp_rate/60))
if((window_size % 2) == 1):
    window_size += 1

class_buffer = np.zeros((window_size, 1)) # init with zero
time_buffer = np.zeros((window_size, 1)) # init with zero
raw_acc_buffer = np.zeros((window_size, 3)) # init with zero accel

window_open = True


##########################################
####     Misc Function Definitions    ####
##########################################

class KeyPressWindow(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

def keyPressed(evt):
    global current_class, window_open, vel
    if(evt.key() == 81): # q
        vel[:] = 0


#########################
####    Qt Section   ####
#########################

app = pg.mkQApp("Plotting Example")
win = KeyPressWindow(show=False, title="Basic plotting examples")
win.sigKeyPress.connect(keyPressed)
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

#wd = pg.GraphicsLayoutWidget(title="Pocket Detect Server")
#layoutgb = QtWidgets.QGridLayout()

#layoutgb.setColumnStretch(0, 20)
#layoutgb.setColumnStretch(1, 20)
#wd.setLayout(layoutgb)

a1 = win.addPlot(row=0, col=0, colspan=1)
a2 = win.addPlot(row=1, col=0, title="Velocity", font_size=300, colspan=1)
a1.setLabel('left', "Acceleration")
a2.setLabel('left', "Velocity")
font=QtGui.QFont()
font.setPixelSize(65)
a1.getAxis("left").label.setFont(font)
a2.getAxis("left").label.setFont(font)

a1.enableAutoRange('xy', True)
a2.enableAutoRange('xy', True)
a1.setXRange(0, window_size, padding=0)
a2.setXRange(0, window_size, padding=0)
# a1.setYRange(-3, 3, padding=0.1)
# a2.setYRange(-3, 3, padding=0.1)
acc_curve1 = a1.plot(raw_acc_buffer[:, 0], pen=pg.mkPen((255, 50, 50), width=3))
acc_curve2 = a2.plot(raw_acc_buffer[:, 1], pen=pg.mkPen((50, 255, 50), width=3))

#layoutgb.addWidget(win, 0, 0, 1, 2)

win.show()

kill_threads = False



accel = np.zeros((len(raw_acc_buffer),))
vel = np.zeros((len(raw_acc_buffer),))
last_time = 0
def process_data():
    global last_time, raw_acc_buffer, kill_threads, accel, vel

    while not kill_threads:
        try:
            l = ser.readline()[:-2].decode('utf-8')
        except UnicodeDecodeError:
            continue

        if(len(l) < 10):
            continue
        try:
            acc = l.split('.')
            acc[0] = acc[0] + '.' + acc[1][:2]
            acc[1] = acc[1][2:] + '.' + acc[2][:2]
            acc[2] = acc[2][2:] + '.' + acc[3][:2]
            acc[3] = 0#acc[3][2:]
            for i in range(len(acc)):
                acc[i] = float(acc[i])
            acc = acc[:-1]
        except IndexError:
            continue

        if(len(acc) == 3):
            acc = np.expand_dims(np.array(acc), axis=1).reshape(1, 3)
            raw_acc_buffer = np.concatenate([acc,  raw_acc_buffer[:-1]])

            # Convert data
            accel = raw_acc_buffer[:, 2] - 1.07

            dt = 1/1000
            vel = np.roll(vel, 1)
            vel[0] = vel[1] + accel[0] * dt

        # Calculate frame rate
        t = time.time() - last_time
        last_time = time.time()
        if t > 0:
            print("FPS: ", 1 / t)


def update():
    global accel, vel

    # Update all curve plots

    acc_curve1.setData(accel)
    acc_curve2.setData(vel)

    time.sleep(0.01)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys, threading

    t1 = threading.Thread(target=process_data)
    #t2 = threading.Thread(target=mouse_listener)
    t1.start()
    #t2.start()

    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'): 
            QtWidgets.QApplication.instance().exec_()

    finally:

        kill_threads = True

        # Stop streaming
        timer.stop()

        t1.join()