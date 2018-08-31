import cv2
import numpy as np
# coding=gbk
from PIL import Image
import numpy as np
# import scipy
#以2008 apple 10k為範例有98頁
#把他們變成3維資料  
#切割成50*400
wc=50
wl=400    
cutw=int(np.floor(1754/wc))
cutl=int(np.floor(1240/wl))
totaldata=[]   
pagecount=[]
for m in range(2008,2017,1):   #  選擇2008~2017年的data當作training data
    
    for i in range(1,9,1):   #讀取個位數
        temp="apple 10k/image/"+str(m) +" d10k/d10k_p00"+str(i)+".png"
        src = cv2.imread(temp)

        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        for j in range(cutw):
            for k in range(cutl):
                cuttemp=gray[wc*j:wc*(j+1),wl*k:wl*(k+1)]
                totaldata.append(cuttemp)
            if k+1==cutl:
                cuttemp=gray[wc*j:wc*(j+1),1240-wl:1240]
                totaldata.append(cuttemp)

    
    for i in range(10,99,1):    #讀取十位數
        temp="apple 10k/image/"+str(m) +" d10k/d10k_p0"+str(i)+".png"
        src = cv2.imread(temp)
        if np.shape(src)==():
            break
        gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        for j in range(cutw):
            for k in range(cutl):
                cuttemp=gray[wc*j:wc*(j+1),wl*k:wl*(k+1)]
                totaldata.append(cuttemp)
            if k+1==cutl:
                cuttemp=gray[wc*j:wc*(j+1),1240-wl:1240]
                totaldata.append(cuttemp)
    (page,width,length)=np.shape(totaldata)
    pagecount.append(page)  #紀錄每次新增完後的頁數，為了之後label方便
#np.shape(totaldata) #確認input


totallabel=[]
#處理output  label###################################
for i in range(pagecount[0]):   #2008年
        totallabel.append(0)   #0代表漲
        
for i in range(pagecount[1]-pagecount[0]):   
        totallabel.append(0)   #0代表漲
        
for i in range(pagecount[2]-pagecount[1]):
        totallabel.append(0)   #0代表漲
        
for i in range(pagecount[3]-pagecount[2]):
        totallabel.append(0)   #0代表漲
        
for i in range(pagecount[4]-pagecount[3]):
        totallabel.append(1)   #1代表跌
        
for i in range(pagecount[5]-pagecount[4]):
        totallabel.append(0)   #0代表漲
        
for i in range(pagecount[6]-pagecount[5]):
        totallabel.append(0)   #0代表漲
        
for i in range(pagecount[7]-pagecount[6]):
        totallabel.append(1)   #1代表跌
        
for i in range(pagecount[8]-pagecount[7]):  #2016年
        totallabel.append(0)   #0代表漲




from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils as kutil
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers import Conv3D,MaxPooling3D


#切成test data
ratio=0.7       #切成多少比例為training data   多少比例為test data
[page,width,length]=np.shape(totaldata)
totaldata=np.array(totaldata)

x_train=totaldata[:int(np.floor(ratio*page)),:,:]
x_test=totaldata[int(np.floor(ratio*page)):,:,:]

totallabel=np.array(totallabel)
y_train_label=totallabel[:int(np.floor(ratio*page)),]
y_test_label=totallabel[int(np.floor(ratio*page)):,]

# 處理output
y_train = kutil.to_categorical(y_train_label)
y_test = kutil.to_categorical(y_test_label)


#處理input
(page1,width,length)=np.shape(x_train)
x_train = np.array(x_train).reshape(page1,width,length,1).astype('float32')
(page2,width,length)=np.shape(x_test)
x_test = np.array(x_test).reshape(page2,width,length,1).astype('float32')


# normalization: mean=0, std=1
for i in range(len(x_train)):
    x=x_train[i]
    m=x.mean()
    s=x.std()
    x_train[i]=(x-m)/s
for i in range(len(x_test)):
    x=x_test[i]
    m=x.mean()
    s=x.std()
    x_test[i]=(x-m)/s



# CNN handwritten character recognition
model = Sequential()
#model.add(Conv2D(input_shape=(50, 400,1),
#        filters=16,kernel_size= (5,5),
#        padding='same',
#        activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(input_shape=(1754, 1240,1),
        filters=1,kernel_size= (1000,1000),
        padding='VALID',
      activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# model fitting with training set
train = model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=10,batch_size=32,verbose=2)


# CNN handwritten character recognition
plt.subplot(1,1,1)
plt.title('Train History')
plt.plot(train.history['acc'],'-o',label='train')
plt.plot(train.history['val_acc'],'-x',label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.xticks(list(range(0,10)))
plt.legend()
plt.show()

# model evaluation with test data set
score = model.evaluate(x_test,y_test)


