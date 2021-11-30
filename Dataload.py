from sklearn.model_selection import train_test_split
import os
import torchvision as tv
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torch as t


class Image_Dataset_Dataload(Dataset):
    def __init__(self, images_dir,  labels_dir, transformImage,transfromlabel,mode=None):
        self.mode=mode
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformImage
        self.transformlabel=transfromlabel


    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):

        global image1
        a = []
        if(self.mode=='train'):
            image_name = train[idx]
        else:
            image_name = validation[idx]
        root = ET.fromstring(open(self.labels_dir +'/'+ image_name+'.XML').read())


        for TopMargin in root.iter('{http://www.loc.gov/standards/alto/ns-v2#}TextLine'):

             if (TopMargin.attrib['RECIPIENT'] == 'False'):
                 TopMargin.attrib.clear()

             else:
                del TopMargin.attrib['ID']
                del TopMargin.attrib['RECIPIENT']
                del TopMargin.attrib['BASELINE']
             a.append(TopMargin.attrib)


        while {} in a:
         a.remove({})

        list_B = []
        for i in range(len(a)):
            list_boundingbox = ((list(map(int, (a[i].values())))))
            list_B.append(list_boundingbox)
        boundingbox_array = np.array(list_B)
        img = Image.open(self.images_dir + '/'+image_name+'.png')
        cv2_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        label = np.zeros_like(cv2_img)

        #label_image = Image.fromarray(label)
        #label_image = cv2.cvtColor(np.asarray(label_image), cv2.COLOR_BGR2GRAY)
        if len(boundingbox_array.shape)==2 :

            a,b=boundingbox_array.shape
            for i in range(a):

                cv2.rectangle(label, (boundingbox_array[i][1], boundingbox_array[i][2]), (
                boundingbox_array[i][1] + boundingbox_array[i][3], boundingbox_array[i][0] + boundingbox_array[i][2]),
                      (1,1,1), -1)
        else:
            pass
            # print(self.images_dir + '/' + image_name + '.png')
            # print(self.labels_dir + '/' + image_name + '.XML')
            # os.remove(self.images_dir + '/' + image_name + '.png')
            # os.remove(self.labels_dir + '/' + image_name + '.XML')

        if self.transformI is not None:

             label = cv2.resize(label,(128,128))
             label=t.tensor(label,dtype=t.float32)
             label = label.unsqueeze(0)
             #label=self.transformlabel(label)

             image1 = self.transformI(cv2_img)
        else:
            label=label.astype(dtype=float)
            image1=cv2_img










        #img = img1.permute(2, 0, 1)
        return image1,label


image_name = []
path = 'D:/dataset'
path1 = 'D:/alto'
for root, dirs, files in os.walk(path):
    for name in files:
        image_name.append(name.split('.')[0])


train, validation = train_test_split(image_name, train_size=0.9,test_size=0.1, random_state=42)


def get_train_dataset():
    #TODO
    transform_train1 = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((128,128)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5],std=[0.5])]
    )
    transform_train2 = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((128, 128),interpolation=Image.NEAREST),
        tv.transforms.ToTensor()
        ]

    )
    return Image_Dataset_Dataload(images_dir=path,labels_dir=path1,transformImage=transform_train1,transfromlabel=transform_train2,mode='train')

def get_validation_dataset():
    #TODO
    transform_test = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((128, 128)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5],std=[0.5])
                                ])
    transform_test2 = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.Resize((128, 128),interpolation=Image.NEAREST),
        # tv.transforms.Grayscale(),
        tv.transforms.ToTensor()])
    return Image_Dataset_Dataload(images_dir=path,labels_dir=path1,transformImage=transform_test,transfromlabel=transform_test2,mode='test')

# transform_test = tv.transforms.Compose([
#         tv.transforms.ToPILImage(),
#         tv.transforms.CenterCrop(96),
#         tv.transforms.ToTensor(),
#         tv.transforms.Normalize(mean=[0.5],std=[0.5])
# ])
# a,b=Image_Dataset_Dataload(images_dir=path,labels_dir=path1,transformImage=None,transfromlabel=None,mode='test').__getitem__(3)
# label=cv2.resize(a,(200,200))
# b=cv2.resize(b,(200,200))
#
# cv2.imshow('a',label)
# cv2.waitKey(0)
# cv2.imshow('b',b)
# cv2.waitKey(0)



