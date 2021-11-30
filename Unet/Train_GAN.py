from venv.Unet.Dataload import get_train_dataset,get_validation_dataset
import numpy as np
import torch.utils.data
import torch
from U_Net_Models import AttU_Net
import torch.nn.functional as F
from venv.Unet.Loss import calc_loss,threshold_predictions_p
from venv.Unet.Metrics import dice_coeff
from venv.Unet.Discriminator import DiscriminatorNet
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
epoch=10
loss_G = []
lossL = []
Loss_D=[]
accuracy=[]
lossL.append(np.inf)
loss_G.append(np.inf)
accuracy.append(np.inf)


train_idx=len(get_train_dataset())
validation_idx=len(get_validation_dataset())
train_loader = torch.utils.data.DataLoader(get_train_dataset(), batch_size=batch_size,shuffle=True
                                      )

valid_loader = torch.utils.data.DataLoader(get_validation_dataset(), batch_size=batch_size
                                            )

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")
def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test
model_test = model_unet(AttU_Net, 1, 1)

def D(model,in_channel=1):
    model_D=model(in_channel)
    return model_D
D=D(DiscriminatorNet,1)

model_test.to(device)
initial_lr = 0.0001
opt_G = torch.optim.Adam(model_test.parameters(), lr=initial_lr)
opt_D=torch.optim.Adam(D.parameters(), lr=initial_lr)
MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, MAX_STEP, eta_min=1e-5)


for i in range(epoch):
    trainG_loss = 0.0
    trainD_loss=0.0
    valid_loss = 0.0
    model_test.train()
    k = 1
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt_D.zero_grad()
        scheduler.step(i)
        lr = scheduler.get_lr()
        #x=Noise
        Fake = model_test(x)
        pred_real=D(x,y)
        ones=torch.ones(pred_real.shape)
        loss_dicriminator_real = torch.nn.BCELoss()(pred_real, ones)
        pred_fake = D(x, Fake)
        zero=torch.zeros(pred_fake.shape)
        loss_dicriminator_fake=torch.nn.BCELoss()(pred_fake,zero)
        loss_dicriminator=loss_dicriminator_fake+loss_dicriminator_real
        Loss_D.append(loss_dicriminator)
        loss_dicriminator.backward()
        opt_D.step()
        opt_G.zero_grad()
        Fake=model_test(x)
        loss_bce=torch.nn.BCELoss()(D(x,Fake),zero)
        loss = torch.nn.L1Loss()(Fake, y)  # Dice_loss Used
        lossG=loss+loss_bce
        y_pred1 = F.sigmoid(Fake)
        y_pred1 = y_pred1.detach().numpy()
        y = y.detach().numpy()
        y_pred1 = threshold_predictions_p(y_pred1)
        accuracy = dice_coeff(y_pred1, y)
        trainG_loss += lossG.item()*x.size(0)
        trainD_loss+=loss_dicriminator.item()*x.size(0)
        lossG.backward()
        opt_G.step()
        #x_size = lossG.item() * x.size(0)
        #writer.add_scalar('train_loss')
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
            accuracy1 = dice_coeff(y_pred1, y)
            valid_loss += (lossL.item())


    # model=torch.load('model_test')
    # model.eval()
    trainG_loss = trainG_loss / train_idx
    trainD_loss=trainD_loss/train_idx
    valid_loss = valid_loss / validation_idx


    print('Epoch: {}/{} \tTrainingG Loss: {:.3f} \tTrainingD Loss: {:.3f}\tValidation Loss: {:.3f}\tTrain_Accuracy: {:.3f}\tValidation_Accuracy: {:.3f}'.format(i + 1, epoch, trainG_loss,trainD_loss,
                                                                                      valid_loss, accuracy,accuracy1))
torch.save(model_test,'model_Attunet_Gan')