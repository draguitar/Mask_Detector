from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from playsound import playsound
import time
from pygame import mixer

mixer.init()
sound = mixer.Sound('voice/alarm_1.wav')

IMAGE = "images"
FACE = "face_detector"
MODEL = "mask_detector.model"
CONFIDENCE = 0.5

def mask_detect_voice():
    localtime = time.localtime()
    result = time.strftime("%Y-%m-%d %I:%M:%S %p", localtime)
    print(result)
    path = os.path.abspath("voice/1.mp3")
    playsound(path)    

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    
    # 透過模型向前傳遞
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    


    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # 取信賴程度>50%
        if confidence > CONFIDENCE:
            # 取得人臉座標框
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")


            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # 影像resize 224*224 符合 MobileNetV2
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # 複數個臉，把座標全部存在locs中
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # 人臉的數量>0，才進行預測
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # 回傳框框的(座標,是否戴口罩)
    return (locs, preds)


prototxtPath = os.path.sep.join([FACE, "deploy.prototxt"])
weightsPath = os.path.sep.join([FACE, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model(MODEL)

print("[INFO] Starting stream ......")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 迴圈視訊串流，反覆執行
while True:
    # 指定畫面 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # 取得座標&是否戴口罩的預測結果
    try:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
        
        for (box, pred) in zip(locs, preds):
            # Bounding box 座標
            (startX, startY, endX, endY) = box
            
            # mask & withoutMaks機率
            (mask, withoutMask) = pred
    
            # 口罩大於沒口罩顯示藍色框
            # 沒口罩大於口罩顯示紅色框
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            if mask < withoutMask:
                sound.play()
            # 將預測用文字顯示在畫面上
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    except Exception as ex:
        print('Running ......')

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



