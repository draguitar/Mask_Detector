# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
# %%
# 資料集路徑
DATASET = 'dataset'

# Learning Rate、ephchs、batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

def main():
    # 圖片存成list，並透過轉成img_to_array轉成ndarray
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(DATASET))
    data = []
    labels = []
    
    
    for imagePath in imagePaths:
    	# 透過資料夾取得Label(with_mask、without_mask)
    	label = imagePath.split(os.path.sep)[-2]
        
    	# 以224 * 224 讀入圖片
    	image = load_img(imagePath, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)
        
    	data.append(image)
    	labels.append(label)
    
    # 資料轉成Numpy array
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    # 將標籤(with_mask、without_mak)，one hot encoder轉換模型可用型態
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    """
      全部以[[0, 1]
            [1, 0]]表示
    """
    # %%
    # 切分dataset 75% for training、25% for testing 建立模型
    (trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.2, stratify=labels, random_state=0)
    
    # 建立數據增強器augmentation
    # 參數說明參考https://keras.io/zh/preprocessing/image/
    aug = ImageDataGenerator(
    	rotation_range=20,
    	zoom_range=0.15,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	shear_range=0.15,
    	horizontal_flip=True,
    	fill_mode="nearest")
    
    # transfer learning MobileNetV2 
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
    	input_tensor=Input(shape=(224, 224, 3)))
    
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    
    # 更換Fully Connected Layer
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    
    for layer in baseModel.layers:
    	layer.trainable = False
    
    
    print("[INFO] Compiling model ......")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    
    print("[INFO] Training ......")
    H = model.fit(
    	aug.flow(trainX, trainY, batch_size=BS),
    	steps_per_epoch=len(trainX) // BS,
    	validation_data=(testX, testY),
    	validation_steps=len(testX) // BS,
    	epochs=EPOCHS)
    
    # 預測testing data
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    
    
    predIdxs = np.argmax(predIdxs, axis=1)
    
    # 混淆矩陣
    print(classification_report(testY.argmax(axis=1), predIdxs,
    	target_names=lb.classes_))
    
    # 儲存模型
    print("[INFO] Saving model ......")
    model.save('mask_detector_2.model', save_format="h5")
    
    # plot the Training loss and Accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('history.png', dpi=150)

if __name__ == '__main__':
    main()
    
