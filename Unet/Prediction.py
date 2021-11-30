import cv2
from venv.Unet.Dataload import get_validation_dataset
import numpy as np
import torch.utils.data
import torch
from venv.Unet.Loss import threshold_predictions_p
from venv.Unet.Metrics import dice_coeff

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')
batch_size=1


valid_loader = torch.utils.data.DataLoader(get_validation_dataset(), batch_size=batch_size
                                            )
model = torch.load('model_Attunet2')
model.eval()
for i in range(5):
     image,label=valid_loader.dataset.__getitem__(i)
     image=image.unsqueeze(0)
     out=model(image)
     image=image.permute(0,2,3,1)
     x1 = out.permute(0, 2, 3, 1)
     x2=x1[0,:,:,:].detach().numpy()
     y11=label[0,:,:]
     y11=y11.detach().numpy()
     image=image[0,:,:,:].detach().numpy()
     y11=cv2.resize(y11,(512,512))
     x2=threshold_predictions_p(x2)
     x2=cv2.resize(x2,(512,512))
     x2=x2.astype(np.uint8)
     image=cv2.resize(image,(512,512))
     count,cnt=cv2.findContours(x2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
     cv2.drawContours(image, count, -1, 255,1)
     a=np.hstack((y11,image))
     cv2.imshow('1',a)
     cv2.waitKey(0)
#cv2.imwrite('Attunet_label and prediction4.png',a*255)


#image=np.hstack((x2,y11))
#cv2.imshow('p',image)
#cv2.imwrite("image.png",image*255)
#cv2.waitKey(0)

accuracy_test=[]
for i in range(20):
    image,label=valid_loader.dataset.__getitem__(i)
    image=image.unsqueeze(0)

    out=model(image)
    x1 = out.permute(0, 2, 3, 1)
    x2=x1[0,:,:,:].detach().numpy()
    y11=label[0,:,:]
    y11=y11.detach().numpy()
    x2=cv2.resize(x2,(512,512))
    y11=cv2.resize(y11,(512,512))
    x2 = threshold_predictions_p(x2)
    accuracy_test.append(dice_coeff(x2,y11))
    #print(accuracy_test)

print('accuracy_test:',sum(accuracy_test)/20)
