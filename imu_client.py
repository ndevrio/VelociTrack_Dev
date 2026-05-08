import serial
import time
import numpy as np
import socket
import pickle

HOST = "127.0.0.1"
PORT = 8083

ser = serial.Serial('COM3', 1000000)

imu_time = time.time()
frame_count = 0

# connect to the server
imu_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
imu_socket.connect((HOST, PORT))

while True:
    try:
        l = ser.readline().decode('utf-8').strip()
        timestamp = time.time()
    except UnicodeDecodeError:
        continue
    
    l = l.split('.')
    
    imu_data = np.zeros((5,))
    imu_data[0] = time.time()
    imu_data[1] = float(l[0] + '.' + l[1][:2])
    imu_data[2] = float(l[1][2:] + '.' + l[2][:2])
    imu_data[3] = float(l[2][2:] + '.' + l[3][:2])
    imu_data[4] = float(l[3][2:])

    # send the data to the server
    imu_socket.sendall(pickle.dumps(imu_data))

    frame_count += 1

    # Calculate frame rate
    if(time.time() - imu_time > 1.0):
        print("FPS: ", frame_count)
        frame_count = 0
        imu_time = time.time()