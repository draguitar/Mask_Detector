
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

import glob

IMAGE = "images"
FACE = "face_detector"
MODEL = "mask_detector.model"
CONFIDENCE = 0.5



# 辨識口罩顏色
def calculate_mask_area(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    color = (0,'')
    # blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    blue_count = (cv2.countNonZero(blue_mask),'BLUE')
    color = (blue_count[0],blue_count[1])
    # Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    red_count = (cv2.countNonZero(red_mask),'RED')
    if (red_count[0]>color[0]):
        color = (red_count[0],red_count[1])
    # Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([150, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    green_count = (cv2.countNonZero(green_mask),'GREEN')
    if (green_count[0]>color[0]):
        color = (green_count[0],green_count[1])
    # White color
    low_white = np.array([0, 0, 230])
    high_white = np.array([360, 10, 255])
    white_mask = cv2.inRange(hsv_frame, low_white, high_white)
    white = cv2.bitwise_and(frame, frame, mask=white_mask)
    white_count = (cv2.countNonZero(white_mask),'WHITE')
    if (white_count[0]>color[0]):
        color = (white_count[0],white_count[1])
    # Black color
    low_black = np.array([0, 0, 0])
    high_black = np.array([360, 255, 15])
    black_mask = cv2.inRange(hsv_frame, low_black, high_black)
    black = cv2.bitwise_and(frame, frame, mask=black_mask)
    black_count = (cv2.countNonZero(black_mask),'BLACK')
    if (black_count[0]>color[0]):
        color = (black_count[0],black_count[1])
        
    return color


def mask_image():
    # 每幾秒撥放一張
    persecond = 5000
    
    # 載入臉部偵測opencv model
    prototxtPath = os.path.sep.join([FACE, "deploy.prototxt"])
    weightsPath = os.path.sep.join([FACE, "res10_300x300_ssd_iter_140000.caffemodel"])
    
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # 載入口罩偵測模型
    model = load_model(MODEL)


    img_list = glob.glob(IMAGE+'/*.*')
    for (idx, img) in enumerate(img_list) :
        image = cv2.imread(img)
        
        orig = image.copy()
        (h, w) = image.shape[:2]
        
        # 圖片前處理，資料標準化
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        
        net.setInput(blob)
        
        detections = net.forward()
        print(detections.shape)
        print(detections.shape[2])
        
        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
            # 取信賴程度>50%
            if confidence > CONFIDENCE:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
    

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
                # 將臉的畫面框起來
                face = image[startY:endY, startX:endX]
                
                mask_color = calculate_mask_area(face)
                # print(mask_color[1])
                # 辨識口罩顏色
                
                
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
    

                (mask, withoutMask) = model.predict(face)[0]
    
                # 戴口罩顯示藍色、未戴口罩顯示紅色
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (255, 0, 0) if label == "Mask" else (0, 0, 255)
    
                # 呈現機率
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
                # 將結果顯示在圖片上
                text = f'{mask_color[1]} - {label}' if mask > withoutMask else f"{label}"
                cv2.putText(image, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
        # 顯示圖片
        cv2.imshow("Output", image)
        k = cv2.waitKey(persecond)
        # Esc: 27
        # Space: 32
        # Enter: 13
        if k in (27,13,32) :
            persecond = 0
            cv2.imshow("Output", image)
        cv2.destroyAllWindows() 
        
    
if __name__ == "__main__":
    mask_image()

# %%