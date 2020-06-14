# -*- coding: utf-8 -*-

import age_gender_service_settings as settings
import redis
import time
import json

import base64
import sys

import cv2
import numpy as np
from wide_resnet import WideResNet

import os
import mylogging
# 初始化日志文件对象
logfolder = './log/app_log'
if not(os.path.exists(logfolder)):
    os.makedirs(logfolder)
    
logfile = mylogging.mylogging(logfolder + "/age_gender_service.log")

'''
### base64编码函数，将numpy array编码为base64
#   a : in shape (batch_size, width, height, channel)
'''
def base64_encode_image(a):
    # base64 encode the input NumPy array
	return base64.b64encode(a).decode("utf-8")

'''
### base64解码函数，将base64 解码为numpy array
#   a : 经过base64_encode_image编码的结果
'''
def base64_decode_image(a, dtype, shape):
	# if this is Python 3, we need the extra step of encoding the
	# serialized NumPy string as a byte object
	if sys.version_info.major == 3:
		a = bytes(a, encoding="utf-8")

	# convert the string to a NumPy array using the supplied data
	# type and target shape
	a = np.frombuffer(base64.decodestring(a), dtype=dtype)  
	a = a.reshape(shape)

	# return the decoded image
	return a


"""
Face detection
"""

class FaceCV(object):
    FACE_DETECTOR = settings.MODEL_PARA['detect_faces_model_path']
    # WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
    AGE_GENDER_PREDICTOR = settings.MODEL_PARA['age_gender_prediction_path']

    """
        Singleton class for face recongnition task
    """
    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        self.model.load_weights(self.AGE_GENDER_PREDICTOR)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=settings.MARGIN, size=settings.FACE_SIZE):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


    def detect_face_one_image(self, frame):
        out_boxes = []
        out_ages = []
        out_genders = []

        # load the face detector model
        face_cascade = cv2.CascadeClassifier(self.FACE_DETECTOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=10,
            minSize=(self.face_size, self.face_size)
        )
        # placeholder for cropped faces
        face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
        print('len(faces) :{}'.format(len(faces)))

        for i, face in enumerate(faces):
            face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
            (x, y, w, h) = cropped
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            out_boxes.append(cropped)
            face_imgs[i, :, :, :] = face_img
            print(face_imgs.shape)
        print("out_boxes : {}".format(out_boxes))

        if len(face_imgs) > 0:
            # predict ages and genders of the detected faces
            results = self.model.predict(face_imgs)
            print("----doing face_imgs----------")

            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, face in enumerate(faces):
            out_ages.append(int(predicted_ages[i]))
            out_genders.append("F" if predicted_genders[i][0] > 0.5 else "M")
            # label = "{}, {}".format(int(predicted_ages[i]),
            #                         "F" if predicted_genders[i][0] > 0.5 else "M")
            # self.draw_label(frame, (face[0], face[1]), label)
        print("out_boxes : {}".format(out_boxes))
        return out_boxes, out_ages, out_genders


'''
### Par-1. 连接到Redis数据库
'''
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

'''
### Par-2. 模型服务主要函数
'''
def detect_process():
    '''
    #1.  加载模型
    '''
    logfile.addinfolog("Loading model...")
    model = FaceCV(depth=settings.NETWORK_DEPTH, width=settings.NETWORK_WIDTH)
    logfile.addinfolog("Model Loaded")

    '''
    #2.  循环，从Redis获取请求数据，模型预测，返回结果到Redis
    '''
    while True:
        '''
        # 从Redis数据库，获取一张图片
        '''
        age_gender_request_list = db.lrange(settings.IMAGE_QUEUE, 0, 0)
        
        imageID = None
        imageNP = None
        for age_gender_request in age_gender_request_list:
            age_gender_request = json.loads(age_gender_request.decode("utf-8"))
            imageID = age_gender_request['id']
            imageNP = base64_decode_image(age_gender_request['image'], settings.IMAGE_DTYPE,
                                          (settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,
                                           settings.IMAGE_CHANS))
            print(imageNP.shape)

        '''
        # 模型处理
        '''
        if imageNP is not None:
            logfile.addinfolog("Start Process Request")
            out_boxes, out_ages, out_genders = model.detect_face_one_image(imageNP)
            logfile.addinfolog("End Process Request")
            predicted_result = {
                'out_boxes': np.array(out_boxes).astype(str).tolist(),
                'out_ages': np.array(out_ages).astype(str).tolist(),
                'out_genders': out_genders
            }
            
            db.set(imageID, json.dumps(predicted_result))
            db.ltrim(settings.IMAGE_QUEUE, 1, -1)

        # 睡眠
        time.sleep(settings.SERVER_SLEEP)

if __name__ == "__main__":
    detect_process()

    #face = FaceCV(depth=settings.NETWORK_DEPTH, width=settings.NETWORK_WIDTH)
    #image_np = cv2.imread('./tww.png')
    #face.detect_face_one_image(image_np)


