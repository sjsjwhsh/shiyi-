import numpy as np
import argparse
import cv2

from tensorflow.keras.models import Sequential

from tkinter import filedialog
from PIL import Image, ImageTk
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

tf.config.run_functions_eagerly(True)
# tf.compat.v1.disable_eager_execution()

# command line argument
# ap = argparse.ArgumentParser()
# ap.add_argument("--mode", help="display")
# a = ap.parse_args()
# mode = a.mode
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="display", default="image")  # 设置默认值为 "image"
a = ap.parse_args()
mode = a.mode


def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

###

##
# Create the model
model = Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(7, activation='softmax'))

# emotions will be displayed on your face from the webcam feed

model.load_weights(
    r'D:\kkss\my_face\model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def image_recognize(img_path, save_path):
    # Find haar cascade to draw bounding box around face
    frame = cv2.imread(img_path) #用 OpenCV 的 imread 函数读取指定路径的图像
    facecasc = cv2.CascadeClassifier(
        r'D:/kkss/my_face/haarcascade_frontalface_default.xml')  #创建人脸检测器，使用 Haar 特征分类器
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    # show the output frame
    cv2.imwrite(save_path, frame)
    return save_path


def video_recognize(v_path, s_path):
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # 指定视频编码方式
    videoWriter = cv2.VideoWriter(s_path, fourcc, fps, (w, h))  # 创建视频写对象
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数

    if v_path == 0:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    else:
        while 1:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if frame is None:
                break
            facecasc = cv2.CascadeClassifier(
                # r'C:/Users/Administrator/Desktop/learn/cloud-acting/face_recognize_code/Python/my_face/haarcascade_frontalface_default.xml')
                r'D:/kkss/my_face/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            videoWriter.write(frame)
            key = cv2.waitKey(1) & 0xFF
    return s_path

# path_ = filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))
# path = path_[0]  # path_为元组，将地址从元组中取出

# si_path=r'C:\Users\Administrator\Desktop\10.jpg'
# image_path=image_recognize(path,si_path)
# img=Image.open(image_path)
# img.show()
