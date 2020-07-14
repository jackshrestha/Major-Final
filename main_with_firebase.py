import numpy as np
import cv2
import csv
import os
import FaceEncoding as facee
import pickle
import sqlite3
import dlib
from os.path import exists
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallervggnet import SmallerVGGNet
from firebase import firebase
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import random
import datetime
from datetime import timedelta


from keras.models import load_model
import imutils

import matplotlib
matplotlib.use("Agg")


cap=cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

firebase_database = firebase.FirebaseApplication("https://attendance-dec88.firebaseio.com/",None)


def Registration():  

    First="Ishwor"
    Last="Shrestha"
    Roll = "73011"
    Faculty="BEX"
    Batch="2073"

    folderPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "dataset/"+Batch+"/"+Faculty+"/"+First+" "+Last)

    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    sampleNum = 0
    while(True):
        ret, img = cap.read() 
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                                    
        dets = detector(img, 1)                                                    
        for i, d in enumerate(dets):                                                
            sampleNum += 1
            cv2.imwrite(folderPath + "/"+"Sample."+ str(sampleNum) + ".jpg",
                        gray[d.top()+10:d.bottom()+10, d.left()+10:d.right()+10])     
            cv2.rectangle(img, (d.left(), d.top())  ,(d.right(), d.bottom()),(0,255,0) ,2) 
                                                                 
        cv2.imshow('frame', img)
        cv2.waitKey(1)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
        elif sampleNum>50:
            break


    cap.release()
    cv2.destroyAllWindows() 

    print("..........*****Training the model*****.........")

    dataset = folderPath

    EPOCHS = 10
    INIT_LR = 1e-3
    BS = 32
    IMAGE_DIMS = (96, 96, 3)

    data = []
    labels = []

    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images("dataset")))
    random.seed(42)
    random.shuffle(imagePaths)


    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        label = Faculty+Roll
        print(label)
        labels.append(label)


    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print("[INFO] data matrix: {:.2f}MB".format(
        data.nbytes / (1024 * 1000.0)))

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.2, random_state=42)

   
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    print("[INFO] compiling model...")
    model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(lb.classes_))
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    print("[INFO] training network...")
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=10,
        #steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)


    print("[INFO] serializing network...")
    model.save('facefeatures_model.h5')
   
    print("[INFO] serializing label binarizer...")
    f = open("labelbin", "wb")
    f.write(pickle.dumps(lb))
    f.close()



    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(folderPath+"/"+First)



#Registration()



def recognition():

    Entry=False
    stop_recording=False
    Entry_time_Recorded=False
    one_time_buffer_run=False
    
    while True:
        date_time=datetime.datetime.now()
        
        detect_face=False
        ret, img = cap.read() 
        font = cv2.FONT_HERSHEY_PLAIN
        image = cv2.resize(img, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        output = img.copy()
        model = load_model("facefeatures_model.h5")

        lb = pickle.loads(open("labelbin", "rb").read())
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        id_code = lb.classes_[idx]
                                        
        dets = detector(img, 1)

        for i, d in enumerate(dets):  
            detect_face=True
            cv2.rectangle(img, (d.left(), d.top())  ,(d.right(), d.bottom()),(0,255,0) ,2)    
            label = "{}: {:.2f}%".format(id_code, proba[idx] * 100)
            output = imutils.resize(output, width=400)
            cv2.putText(img, label,(d.left(),d.top()), font, 1,(255,0,0),1) 
            print(label)
            time=date_time.strftime("%H:%M:%S")
            date=str(date_time.date())
            if(Entry==False):
                firebase_database.put("Attendence/Students/"+id_code+"/"+date+"/","Entry Time",time)
                print("Entry:"+time)
                Entry=True 
                Entry_time_Recorded=True

       
        if(detect_face==False and stop_recording==False and Entry_time_Recorded==True):
          
            time=date_time.strftime("%H:%M:%S")
            date=str(date_time.date())
            current_time= str(datetime.datetime.now())
            if(one_time_buffer_run==False):
                buffer_time =  str(datetime.datetime.now() + timedelta(seconds=30))
                print(buffer_time)
                one_time_buffer_run=True

            #print(current_time + "=" + buffer_time)
          
            #if(current_time==buffer_time):  
                firebase_database.put("Attendence/Students/"+id_code+"/"+date+"/","Exit Time",time)
                print("Exit:"+time)
                stop_recording=True
        cv2.imshow('frame', img)                                                   
       # cv2.waitKey(1)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 
    

recognition()


