# Server (server.py)
import time
import socket
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


### Connect to socket server
host = ''  # Listen on all available interfaces
port = 8080  # Choose a port number

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
server_socket.settimeout(50)

print(f"Server listening on {host}:{port}")
    

window_size = 210
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
    global current_class, window_open, vel, depth
    if(evt.key() == QtCore.Qt.Key_Space): # q
        vel[:] = 0
        depth[:] = 0


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
a2 = win.addPlot(row=1, col=0, colspan=1)
a3 = win.addPlot(row=2, col=0, colspan=1)
a1.setLabel('left', "Depth")
a2.setLabel('left', "Velocity")
a3.setLabel('left', "Acceleration")
font=QtGui.QFont()
font.setPixelSize(65)
a1.getAxis("left").label.setFont(font)
a2.getAxis("left").label.setFont(font)
a3.getAxis("left").label.setFont(font)

a1.enableAutoRange('xy', True)
a2.enableAutoRange('xy', True)
a3.enableAutoRange('xy', True)
a1.setXRange(0, window_size, padding=0)
a2.setXRange(0, window_size, padding=0)
a3.setXRange(0, window_size, padding=0)
a1.setYRange(0, .6, padding=0.1)
a2.setYRange(-.01, .01, padding=0.1)
a3.setYRange(-.01, .01, padding=0.1)
acc_curve1 = a1.plot(raw_acc_buffer[:, 0], pen=pg.mkPen((255, 255, 255), width=3))
acc_curve2 = a2.plot(raw_acc_buffer[:, 1], pen=pg.mkPen((100, 255, 100), width=3))
acc_curve3 = a3.plot(raw_acc_buffer[:, 1], pen=pg.mkPen((255, 100, 100), width=3))

#layoutgb.addWidget(win, 0, 0, 1, 2)

win.show()

kill_threads = False



last_time = 0
def process_data():
    global kill_threads, last_time, raw_acc_buffer, server_socket
    
    while True:
        try:
            client_socket, client_address = server_socket.accept()
        except TimeoutError:
            continue
        print(f"Connection from {client_address}")

        while not kill_threads:
            data = client_socket.recv(1024)
            if not data:
                break
            message = data.decode()
            
            
            acc = message.split(' ')
            
            try:
                # print(acc, np.array(acc).astype(float))
                
                if(len(acc) == 3):
                    acc = np.expand_dims(np.array(acc).astype(float), axis=1).reshape(1, 3)
                    raw_acc_buffer = np.concatenate([raw_acc_buffer[1:], acc])

                # Calculate frame rate
                t = time.time() - last_time
                last_time = time.time()
                if t > 0:
                    print("FPS: ", 1 / t)
            except ValueError:
                continue

        client_socket.close()
        print(f"Connection closed with {client_address}")


def update():
    global raw_acc_buffer

    # Update all curve plots
    
    # acc_curve1.setData(raw_acc_buffer[:, 0])
    # acc_curve2.setData(raw_acc_buffer[:, 1])
    # acc_curve3.setData(raw_acc_buffer[:, 2])
    
    depth = raw_acc_buffer[:, 2]
    
    rval = 1
    vel = depth - np.roll(depth, rval)
    acc = vel - np.roll(vel, rval)
    
    acc_curve1.setData(depth)
    acc_curve2.setData(vel)
    acc_curve3.setData(acc)

    time.sleep(0.01)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys, threading

    t1 = threading.Thread(target=process_data)
    t1.start()

    try:
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    finally:

        kill_threads = True

        # Stop streaming
        timer.stop()