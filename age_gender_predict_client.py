# -*- coding: utf-8 -*-
import redis
import uuid
import time
import json
import numpy as np
from PIL import Image
import cv2
import sys
import os

# import self tool
sys.path.append(os.path.split(os.path.realpath(__file__))[0])
import helpers
import age_gender_service_settings as settings


db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess_image_pd(image_np):
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    boxed_image = letterbox_image(image, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
        
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    
    return image_data

def age_gender_predict(image_np):
    
    data = {"success": False}
    
    orig_height, orig_width, _ = image_np.shape
    '''
    # 1. Compose and send the request to db
    '''
    #image_np = preprocess_image_pd(image_np)
    #image_np = image_np.copy(order="C")
    image_id = str(uuid.uuid4())
    input_data = {'id': image_id, 'image': helpers.base64_encode_image(image_np)}
    
    db.rpush(settings.IMAGE_QUEUE, json.dumps(input_data))
    
    '''
    # 2. Loop the db to get the response
    '''
    while True:
        output_data = db.get(image_id)
        if output_data is not None:
            output_data = output_data.decode("utf-8")
            output_data = json.loads(output_data)
            output_data['out_boxes'] = np.array(output_data['out_boxes']).astype('float32')
            output_data['out_ages'] = np.array(output_data['out_ages']).astype('int64')
            output_data['out_genders'] = np.array(output_data['out_genders'])
            data["predictions"] = output_data
            
            db.delete(image_id)
            break
        
        time.sleep(settings.CLIENT_SLEEP)
        
    data["success"] = True
    
    return data

'''
### PD Service Test

import cv2
img_file = 'test_images/2610.JPG'
image_np = cv2.imread(img_file)

data = age_gender_predict(image_np)
if data['success']:
    out_boxes = data['predictions']['out_boxes']
    out_scores = data['predictions']['out_scores']
    out_classes = data['predictions']['out_classes']
    
    for id in range(len(out_classes)):
        y1, x1, y2, x2 = out_boxes[id]
        cv2.rectangle(image_np, (x1,y1), (x2, y2), (0,0,255), 2)
    
    cv2.imwrite('test_images/results.JPG', image_np)
    
print(data)

'''
import cv2

# camera_source = './videos/dongdong.mp4'
camera_source = 0
cap = cv2.VideoCapture(camera_source)
cap.set(3, settings.IMAGE_WIDTH)
cap.set(4, settings.IMAGE_HEIGHT)

print("---------start reading video---------")
while True:
    res, image_np = cap.read()
    if not(res):
        break
    print("----------read sucess--------------")
    im_height, im_width, _ = image_np.shape
    data = age_gender_predict(image_np)
    print("data : {} ".format(data))
    # cv2.imshow('orig', image_np)
    if data['success']:
        print("---------predict sucess-----------")
        out_boxes = data['predictions']['out_boxes']
        out_ages = data['predictions']['out_ages']
        out_genders = data['predictions']['out_genders']
        
        for id in range(len(out_boxes)):
            (x, y, w, h) = out_boxes[id][0], out_boxes[id][1], out_boxes[id][2], out_boxes[id][3]
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(image_np, str(out_ages[id]), (int(x+w/2-40),int(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image_np, str(out_genders[id]), (int(x+w/2-40)+40,int(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('test', image_np)
    
    # key = cv2.waitKey(5)

    butt = cv2.waitKey(10) & 0xFF
    if butt == ord('q'):
        break

cap.release()
