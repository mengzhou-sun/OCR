from sklearn.model_selection import train_test_split
import os
import numpy as np
import csv
import cv2
from PIL import Image
from torch.utils.data import Dataset

class Image_Dataset_Dataload(Dataset):
    def __init__(self, images_dir, labels_dir,label_len,list_name):
        self.image_dir=images_dir
        self.labels_dir=labels_dir
        self.label_len=label_len
        self.list_name=list_name


    def __len__(self):

        return len(self.list_name)

    def __getitem__(self, idx):
        global outImg
        IMG_HEIGHT = 64
        IMG_WIDTH = 1407

        image_name=self.list_name[idx]
        img=Image.open(self.image_dir + '/'+image_name+'.png')
        img=np.array(img)

        img_width=img.shape[1]


        if img_width < IMG_WIDTH:
            outImg=cv2.copyMakeBorder(img,0,0,0,IMG_WIDTH-img_width,cv2.BORDER_CONSTANT,value=(0,0,0))


        else:
            outImg=img
        #     outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH,3), dtype='uint8')
        #
        #     outImg[:, :img_width] = img
        outImg = outImg / 255.  # float64
        outImg = outImg.astype('float32')
        with open (self.label_len,'r') as db00:
            reader = csv.reader(db00)
            for index,rows in enumerate(reader):
                 if index==idx+1:
                     width=rows


        #out_I=cv2.resize(img,(64,256), interpolation=cv2.INTER_CUBIC)
        with open (self.labels_dir,'r') as db01:
            reader=csv.reader(db01)
            for index,rows in enumerate(reader):
                if index==idx+1:
                    row=rows

        return outImg,row,width
image_name=[]
path='D:/newl'
path1='labe3.csv'
path3='labe3_len.csv'
for root, dirs, files in os.walk(path):
    for name in files:
        image_name.append(name.split('.')[0])
train, validation = train_test_split(image_name, train_size=0.8,test_size=0.2, random_state=42)

def get_train_dataset():
    return Image_Dataset_Dataload(images_dir=path, labels_dir=path1, label_len=path3,list_name=train)

def get_validation_dataset():
    return Image_Dataset_Dataload(images_dir=path, labels_dir=path1,label_len=path3, list_name=validation)

# for i in  range(24):
#     a,b,c,d=Image_Dataset_Dataload(images_dir=path, labels_dir=path1, label_len=path3,list_name=validation).__getitem__(i)
#
#     print(c)
#     if c==1407:
#         a=a*255
#         a=a.astype(np.uint8)
#         im = Image.fromarray(a)
#         im.show()




# print(a)
# cv2.imshow('a',np.array(b))
# cv2.waitKey(0)
# print(c)