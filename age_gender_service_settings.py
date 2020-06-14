# -*- coding: utf-8 -*-
import numpy as np
#1. Redis服务器地址、端口和数据库信息
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

#2. 模型输入图片格式参数 
FACE_SIZE = 64
MARGIN = 40

NETWORK_DEPTH = 16
NETWORK_WIDTH = 8

##==================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CHANS = 3
IMAGE_DTYPE = np.uint8

#3. 队列参数
IMAGE_QUEUE = "age_gender_predict_queue_000001"
BATCH_SIZE = 1

#4. 睡眠参数
SERVER_SLEEP = 0.01
CLIENT_SLEEP = 0.01

#5. 模型参数
MODEL_PARA = {
    "detect_faces_model_path": 'models/detect_faces_model/haarcascade_frontalface_alt.xml',
    "age_gender_prediction_path": 'models/pretrained_age_gender_model/age_gender_weights.18-4.06.hdf5',
    "model_image_size" : (IMAGE_HEIGHT, IMAGE_WIDTH),
    "gpu_num" : 1,
}
gpu_memory_fraction = 0.125
