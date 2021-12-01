import torchvision.models as models
import torch.nn
vgg19=models.vgg19_bn(pretrained=True)
for param in vgg19.parameters():
   new=torch.nn.Sequential(*list(vgg19.children())[:-2])

# a=torch.rand(1,3,32,128)
# print(new(a).shape)
def vgg19(in_data):
    vgg19 = models.vgg19_bn(pretrained=True)
    for param in vgg19.parameters():
      new=torch.nn.Sequential(*list(vgg19.children())[:-2])


    return new(in_data)