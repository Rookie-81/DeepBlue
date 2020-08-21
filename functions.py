import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



############################################################
def importdataset(path):
    columns = ['Center', 'Left', 'Right','Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'),names =columns)
    print(data.head())
    return data

############################################################

def DataBalancing(data,display = True):
    nBins = 31
    samplePerBin = 1500
    histogram , bins =np.histogram(data['Steering'],nBins)
    print(bins)
    if display:
        center = (bins[:-1]+ bins[1:])*0.5
        print(center)
        plt.bar(center,histogram,width = 0.06)
        plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show()

    removeIndexList = []
    for j in range(nBins):
        binDatalist =[]
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDatalist.append(i)
        binDatalist= shuffle(binDatalist)
        binDatalist= binDatalist[samplePerBin:]
        removeIndexList.extend(binDatalist)
    print('Removed Images: ',len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print('Remaining Images: ',len(data))


    if display:
        histogram, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center,histogram,width = 0.06)
        plt.plot((-1,1),(samplePerBin,samplePerBin))
        plt.show()

    return data

########################################################
def DataLoading(path,data):
    imgpath = []
    steering = []

    for i in range(len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imgpath.append(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imgpath = np.asarray(imgpath)
    steering = np.asarray(steering)
    return imgpath,steering

############################################################


def augment(imgpath,steering):
    img = mpimg.imread(imgpath)


    if np.random.rand()< 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)
    if np.random.rand()< 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    if np.random.rand()< 0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)
    if np.random.rand()< 0.5:
        img = cv2.flip(img,1)
        steering = -steering



    return steering,img


def preprocessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img


def batchGen(imgpath,steeringList,batchsize,trainflag):
    while True:
        imgbatch = []
        steeringbatch = []

        for i in range(batchsize):
            index=random.randint(0,len(imgpath)-1)
            if trainflag:
                img , steering = augment(imgpath[index],steeringList[index])
            else:
                img = mpimg.imread(imgpath[index])
                steering = steeringList[index]
                img = preprocessing(img)
                imgbatch.append(img)
                steeringbatch.append(steering)
        yield (np.asarray(imgbatch),np.asarray(steeringbatch))

def modelcreation:
    model = Sequential










