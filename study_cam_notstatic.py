#coding=utf-8
import cv2
import sys
import numpy as np
import mvsdk
import time
import platform
import socket
import mediapipe as mp
from mediapipe.tasks import python
import threading
from collections import deque
import winsound

mp_hands = mp.solutions.hands

labels = {"0": "Hand"}

kill_threads = False
sound_file_path = "beep.wav"



def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, percent, maxwh=None):
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    result = cv2.resize(image, (width, height), interpolation = cv2.INTER_LINEAR) # INTER_CUBIC

    # image = cv2.pyrUp(image) # 2x
    # image = cv2.pyrUp(image) # 4x
    # result = cv2.pyrUp(image) # 8x
    return result, percent

def invariant_match_template(rgbimage, rgbtemplate, method, matched_thresh, rot_range, rot_interval, scale_range, scale_interval):
    """
    rgbimage: RGB image where the search is running.
    rgbtemplate: RGB searched template. It must be not greater than the source image and have the same data type.
    method: [String] Parameter specifying the comparison method
    matched_thresh: [Float] Setting threshold of matched results(0~1).
    rot_range: [Integer] Array of range of rotation angle in degrees. Example: [0,360]
    rot_interval: [Integer] Interval of traversing the range of rotation angle in degrees.
    scale_range: [Integer] Array of range of scaling in percentage. Example: [50,200]
    scale_interval: [Integer] Interval of traversing the range of scaling in percentage.

    Returns: List of satisfied matched points in format [[point.x, point.y], angle, scale].
    """
    img_gray = rgbimage
    template_gray = rgbtemplate
    image_maxwh = img_gray.shape * 10
    height, width = template_gray.shape
    all_points = []
    next_angle = 0
    next_scale = 400.0
    
    scaled_template_gray, actual_scale = scale_image(template_gray, next_scale)
    scaled_img_gray, _ = scale_image(img_gray, next_scale)
    
    # scaled_template_gray, actual_scale = template_gray, 1.0
    # scaled_img_gray = img_gray
    
    matched_points = cv2.matchTemplate(scaled_img_gray, scaled_template_gray, cv2.TM_CCOEFF_NORMED) 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched_points)
    
    max_loc = (max_loc[0] / (next_scale / 100.0), max_loc[1] / (next_scale / 100.0))
    
    if max_val >= matched_thresh:
        all_points.append([max_loc, next_angle, actual_scale, max_val])
    all_points = sorted(all_points, key=lambda x: -x[3])
    
    critical_val = 0.6
    if(max_val > critical_val):
        if(len(all_points) > 0):
            return all_points[0]
        else:
            return [[0, 0], 0, 1, 1]
    else:
        return -1


class CameraCaptureThread(threading.Thread):
    def __init__(self, batch_size=5, dev_info=None):
        threading.Thread.__init__(self)
        
        self.pFrameBuffer = 0
        
        # 打开相机
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(dev_info, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(self.hCamera)
        
        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetGain(self.hCamera, 8, 8, 8)
        mvsdk.CameraSetExposureTime(self.hCamera, 800)
        
        # mvsdk.CameraSetMirror(self.hCamera, 0, 1)
        # mvsdk.CameraSetMirror(self.hCamera, 1, 1)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # 设置采集回调函数
        mvsdk.CameraSetCallbackFunction(self.hCamera, self.GrabCallback, 0)
        
        # Create a queue to store frames
        self.frame_queue = deque(maxlen=batch_size)
        self.running = True
        self.lock = threading.Lock()
        self.frame_count = 0
        
    def run(self):
        while self.running:
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                self.running = False
        
        # 关闭相机
        mvsdk.CameraUnInit(self.hCamera)

        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)
    def get_frame(self):
        if self.get_number_in_queue() > 0:
            return self.frame_queue.popleft()  # Pop and return the oldest frame
        else:
            return None

    def get_buffer(self):
        return self.frame_queue
    
    def get_number_in_queue(self):
        return len(self.frame_queue)

    def stop(self):
        self.running = False
        
    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
    def GrabCallback(self, hCamera, pRawData, pFrameHead, pContext):
        #############################
        ### Receive camera frame ####
        #############################
        FrameHead = pFrameHead[0]
        pFrameBuffer = self.pFrameBuffer

        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        if platform.system() == "Windows":
            mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8).copy() # copy here so buffer doesn't get overwritten when in other thread
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
        
        self.frame_queue.append(frame)
        self.frame_count += 1
        
        # # Convert frame to RGB and increment frame count in one step
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # frame = frame.astype(np.float32, copy=False)  # In-place conversion to float32
        # with self.lock:
        #     self.batched_frames[self.frame_count % batch_size] = frame
        #     self.frame_count += 1
        
        
class App(object):
    def __init__(self):
        super(App, self).__init__()
        
        # Store app info
        self.quit = False
        self.frame = np.zeros((480, 640), dtype=np.uint8)
        
        #continue code from above
        self.DEBUG = sys.argv[2].lower() == 'debug' if len(sys.argv) > 1 else False

        # Socket setup
        self.HOST = "127.0.0.1"
        self.PORT = 8081 + int(sys.argv[1])

        self.send_socket = None
        # if not self.DEBUG:
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.send_socket.connect((self.HOST, self.PORT))
            
        self.WIDTH = 640
        self.HEIGHT = 480
        self.TEMPLATE_SIZE = 40.0 # Lowering this lowers the FPS
        
        self.REINIT_LANDMARKS_COUNT = 30 #80#2400 #180 # Can change this higher to get faster FPS at the expense of worse tracking
        self.DISTANCE_THRESHOLD = 2
        # self.DISTANCE_THRESHOLD = 10


        self.previous_index_finger_tip = [0.0, 0.0]
        self.previous_index_finger_tip_mp = []
        self.mp_did_reset = False
        self.hand_tracked = False
        self.frame_timestamp = 0
        self.frame_updated = False
        
        self.s_num_frames = 0
        self.s_start_time = 0
        self.reset_counter = 0
        self.last_tip_x = 0
        self.last_tip_y = 0
        
        self.bbox = (0, 0, self.TEMPLATE_SIZE, self.TEMPLATE_SIZE)
        self.template = None
        self.region = None
        
        # Mediapipe setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the live stream mode:
        options = HandLandmarkerOptions(
            # base_options=BaseOptions(model_asset_path='./hand_landmarker.task', delegate=python.BaseOptions.Delegate.GPU),
            base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=0.01,
            min_hand_presence_confidence=0.01,
            min_tracking_confidence=0.1,
            num_hands=1)

        # self.landmarker = HandLandmarker.create_from_options(options)

        self.landmarker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.1, min_tracking_confidence=0.1, model_complexity=1, static_image_mode=False)
        
        # Tracker setup
        self.tracker = None
        
    
    def unsharp_mask(self, img, blur_size = (9,9), imgWeight = 1.5, gaussianWeight = -0.5):
        gaussian = cv2.GaussianBlur(img, (5,5), 0)
        return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)


    def main(self):
        # 枚举相机
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(sys.argv[1])
        DevInfo = DevList[i]
        self.PORT += i

        # Create camera capture thread
        camera_capture_thread = CameraCaptureThread(batch_size=50, dev_info=DevInfo)
        camera_capture_thread.start()
        
        self.quit = False
        
        play_mode = 0

        while not self.quit:
            try:
                # get items in queue
                if camera_capture_thread.get_number_in_queue() == 0:
                    time.sleep(0.00001)
                    continue
                else:
                    self.frame = camera_capture_thread.get_frame()
                    self.frame *= 3
                #### Frame processing
                self.frame_updated = True
                self.frame_timestamp = time.time()
                        
                sframe = self.frame.copy()

                ###########################################################
                ### Get a new fingertip tracking result from MediaPipe ####
                ###########################################################
                if self.previous_index_finger_tip == [] or self.s_num_frames % self.REINIT_LANDMARKS_COUNT == 0:
                    # blur the gray frame
                    sframe = sframe[:, :, 0]
                    # blur whole frame
                    sframe = cv2.GaussianBlur(sframe, (9,9), 0)

                    # equalize the histogram
                    # sframe = cv2.equalizeHist(sframe)
                    mp_frame = cv2.cvtColor(sframe, cv2.COLOR_GRAY2RGB)
                    # blur only mediapipe frame
                    # mp_frame = cv2.GaussianBlur(mp_frame, (9,9), 0)
                    results = self.landmarker.process(mp_frame)
                    
                    # print("mediapipe reset")

                    ### If no tracking result is found
                    if not results.multi_hand_landmarks:
                        self.mp_did_reset = True
                    ### If a tracking result is found
                    else:
                        self.previous_index_finger_tip_mp = [results.multi_hand_landmarks[0].landmark[8].x * sframe.shape[1], results.multi_hand_landmarks[0].landmark[8].y * sframe.shape[0]]
                        
                        distance = np.sqrt((self.previous_index_finger_tip_mp[0] - self.previous_index_finger_tip[0])**2 + (self.previous_index_finger_tip_mp[1] - self.previous_index_finger_tip[1])**2)
                        if distance > self.DISTANCE_THRESHOLD:
                            previous_bbox = np.array([(self.previous_index_finger_tip_mp[0] - self.TEMPLATE_SIZE / 2), (self.previous_index_finger_tip_mp[1] - self.TEMPLATE_SIZE / 2), (self.TEMPLATE_SIZE), (self.TEMPLATE_SIZE)])

                            self.previous_index_finger_tip = self.previous_index_finger_tip_mp
                            self.bbox = previous_bbox
                            
                            # Initialize the template
                            self.template = bbox_to_sobel(sframe, self.bbox)
                            
                ############################################################ 
                ### Use previous result with an update from the tracker ####
                ############################################################
                else:
                    pad = 5
                    region_bbox = np.array([self.bbox[0]-pad, self.bbox[1]-pad, self.bbox[2]+pad*2, self.bbox[3]+pad*2])
                    if(region_bbox[0] < 0):
                        region_bbox[2] = region_bbox[2] + region_bbox[0]
                        region_bbox[0] = 0
                    if(region_bbox[1] < 0):
                        region_bbox[3] = region_bbox[3] + region_bbox[1]
                        region_bbox[1] = 0
                    if(region_bbox[0] + region_bbox[2] >= sframe.shape[1]):
                        region_bbox[2] = sframe.shape[1] - region_bbox[0] - 1
                    if(region_bbox[1] + region_bbox[3] >= sframe.shape[0]):
                        region_bbox[3] = sframe.shape[0] - region_bbox[1] - 1
                    region = bbox_to_sobel(sframe, region_bbox)
                    
                    # Perform template matching
                    try:
                        invariant_result = invariant_match_template(rgbimage=region, rgbtemplate=self.template, method="TM_CCOEFF_NORMED", matched_thresh=0.5, rot_range=[-10,20], rot_interval=10, scale_range=[90,110], scale_interval=10)
                        
                        # Check to see if matching was successful
                        if(invariant_result == -1):
                            raise ValueError
                        
                        if(play_mode == 1):
                            # sound_p.terminate()
                            winsound.PlaySound(None, winsound.SND_PURGE)
                            play_mode = 0
                            self.hand_tracked = True
                        
                        # Get the location of the best match
                        max_val, max_loc = invariant_result[3], invariant_result[0]
                        top_left = (max_loc[0] + region_bbox[0], max_loc[1] + region_bbox[1])
                        previous_bbox = np.array([top_left[0], top_left[1], self.bbox[2], self.bbox[3]])
                        
                        self.bbox = previous_bbox
                        self.previous_index_finger_tip = [previous_bbox[0] + self.TEMPLATE_SIZE / 2, previous_bbox[1] + self.TEMPLATE_SIZE / 2]
                    except (AttributeError, cv2.error, ValueError):
                        if(play_mode == 0):
                            winsound.PlaySound(sound_file_path, winsound.SND_ASYNC)
                            play_mode = 1
                            self.hand_tracked = False
                        pass

                #############################
                ### Send data over socket ###
                #############################
                if self.previous_index_finger_tip != []:
                    tip_x, tip_y = self.previous_index_finger_tip[1], self.previous_index_finger_tip[0]
                    
                    ### Send x and y and whether the hand was reinitialized and timestamp
                    message = f"{self.frame_timestamp},{tip_x},{tip_y},{self.mp_did_reset},{self.hand_tracked}" # Mediapipe tracking, template match found
                    self.send_socket.send(message.encode()) 
                    
                    self.last_tip_x, self.last_tip_y = tip_x, tip_y
                self.mp_did_reset = False
                
                ##########################################################################
                ### Display the circle on the image and show both in the OpenCV window ###
                ##########################################################################
                if (self.DEBUG and self.s_num_frames % 30 == 0):
                    try:
                        sobel_frame = bbox_to_sobel(sframe, self.bbox)
                        
                        # display the images together
                        sframe[int(self.bbox[1]):int(self.bbox[1])+int(self.bbox[3]), int(self.bbox[0]):int(self.bbox[0])+int(self.bbox[2])] = sobel_frame

                        cv2.rectangle(sframe, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[0])+int(self.bbox[2]), int(self.bbox[1])+int(self.bbox[3])), (255, 0, 255), 1)

                        try:
                            # draw the previous fingertip
                            cv2.circle(sframe, (int(self.previous_index_finger_tip[0]), int(self.previous_index_finger_tip[1])), 10, (255, 0, 255), 1)
                            sframe = np.rot90(sframe, k=2)
                            cv2.imshow("img", sframe)
                        except cv2.error:
                            pass

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            camera_capture_thread.stop()
                            self.quit = True
                    except ValueError:
                        pass
                
                
                ### Print FPS count
                self.s_num_frames += 1
                if((time.time() - self.s_start_time) > 1):
                    print("FPS:\t", self.s_num_frames)
                    self.s_start_time = time.time()
                    self.s_num_frames = 0
            except KeyboardInterrupt:
                camera_capture_thread.stop()
                self.quit = True 
   

def bbox_to_sobel(frame, region_bbox):
    frame_box = frame[int(region_bbox[1]):int(region_bbox[1]+region_bbox[3]), int(region_bbox[0]):int(region_bbox[0]+region_bbox[2])]
    output = frame_box.copy()

    return output
     

def main():
    global kill_threads
    
    try:
        app = App()
        app.main()
        
    finally:
        cv2.destroyAllWindows()
        kill_threads = True

main()