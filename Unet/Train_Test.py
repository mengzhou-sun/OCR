import cv2
from Dataload import get_train_dataset,get_validation_dataset
import numpy as np
import torch.utils.data
import torch
from U_Net_Models import U_Net
import torch.nn.functional as F
from Loss import calc_loss, dice_loss,threshold_predictions_p
from Metrics import  accuracy_score ,dice_coeff
pin_memory = False

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

if train_on_gpu:
    pin_memory = True
batch_size=4
valid_loss_min = np.Inf
epoch=30
lossT = []
lossL = []
accuracy=[]
lossL.append(np.inf)
lossT.append(np.inf)
accuracy.append(np.inf)
train_loader = torch.utils.data.DataLoader(get_train_dataset(), batch_size=batch_size,
                                            shuffle=True)
valid_loader = torch.utils.data.DataLoader(get_validation_dataset(), batch_size=batch_size,shuffle=False
                                            )
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test
model_test = model_unet(U_Net, 1, 1)
model_test.to(device)
initial_lr = 0.0001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)
MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
for i in range(epoch):
    train_loss = 0.0
    valid_loss = 0.0
    model_test.train()
    k = 1
    for x, y in train_loader:
        """
        x1=x.permute(0,2,3,1)
        x2=x1[0,:,:,:].detach().numpy()
        y11=y[0,:,:]
        y11=y11.permute(1,2,0)
        #
        print(y11.detach().numpy().shape)
        cv2.imshow('a',x2)
        cv2.waitKey(0)
        cv2.imshow('p',y11.detach().numpy())
        cv2.waitKey(0)

        """
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        scheduler.step(i)
        lr = scheduler.get_lr()
        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)  # Dice_loss Used
        y_pred1 = F.sigmoid(y_pred)
        y_pred1 = y_pred1.detach().numpy()
        y = y.detach().numpy()

        y_pred1 = threshold_predictions_p(y_pred1)
        accuracy = accuracy_score(y_pred1, y)
        train_loss += lossT.item()
        lossT.backward()
        opt.step()
        k = 2
    model_test.eval()
    torch.no_grad()
    for x1, y1 in valid_loader:

            x1, y1 = x1.to(device), y1.to(device)

            y_pred1 = model_test(x1)

            lossL = calc_loss(y_pred1, y1)


            y_pred1 = F.sigmoid(y_pred1)
            y_pred1 = y_pred1.detach().numpy()
            y = y1.detach().numpy()
            y_pred1 = threshold_predictions_p(y_pred1)
            accuracy1 = accuracy_score(y_pred1, y)
            # https: // gitlab.cs.fau.de /
            # qy33coty

            valid_loss += lossL.item()




    print('Epoch: {}/{} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f}\tTrain_Accuracy: {:.3f}\tValidation_Accuracy: {:.3f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss, accuracy,accuracy1))
