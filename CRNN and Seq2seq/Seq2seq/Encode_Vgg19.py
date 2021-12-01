#VGG19+GRU
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from VGG19 import vgg19

DROP_OUT = False
LSTM = False
SUM_UP = True

class Encoder(nn.Module):
    def __init__(self, hidden_size, height, width, bgru):
        super().__init__()
        self.hidden_size = hidden_size

        self.height = height
        self.width = width
        self.bi = bgru
        self.n_layers = 2
        self.dropout = 0.5
        self.enc_out_merge = lambda x: x[:, :, :x.shape[-1] // 2] + x[:, :, x.shape[-1] // 2:]

        if bgru==True:
           self.rnn = nn.GRU(self.height // 32 * 512, self.hidden_size, self.n_layers, dropout=self.dropout,
                       bidirectional=True)
#input_size=512
    """
    def __map_to_sequence(self, input_tensor):
          # H of the feature map must equal to 1 1*512*16*1 batch_size,channal,width,height
        return torch.squeeze(input_tensor)
    """
    def forward(self, in_daten,lenth,hidden=None):
        batch_size=in_daten.shape[0]
        #print('original image:',in_daten.shape)#torch.Size([4, 256, 64, 3])
        out=vgg19(in_daten.permute(0,3,1,2))
        #print('nach vgg19 shape:',out.shape)#torch.Size([4, 512, 8, 2])
        out = out.permute(3, 0, 2, 1)
        out=out.contiguous()
        out=out.reshape(4,1024,69)

        #out = out.view(-1, batch_size, (((((self.height - 2) // 2) - 2) // 2 - 2 - 2 - 2) // 2) * 512)
        #out = out.view(-1, batch_size, self.height // 16 * 512)
        #print('die input der rnn:',out.shape)#torch.Size([4, 4, 2048])


        width = out.shape[0]
        src_len = lenth.numpy() * (width / self.width)
        src_len=src_len+0.999
        src_len=src_len.astype('int')
        src_len=torch.Tensor(src_len)
        #print('src_length:',src_len)#tensor([253., 228., 198.,  37.])


        out=out.permute(2,0,1)
        out = pack_padded_sequence(out, src_len.clamp(max=4),batch_first=False)


        #out1 = pack_padded_sequence(out, src_len.clamp(max=4),batch_first=False)



        output, hidden = self.rnn(out, hidden)

        output, output_len = pad_packed_sequence(output,batch_first=False)
        output = self.enc_out_merge(output)

        odd_idx = [1, 3, 5, 7, 9, 11]
        hidden_idx = odd_idx[:self.n_layers]
        final_hidden = hidden[hidden_idx]

        return output, final_hidden
"""
    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / self.width)
        lens2 = lens2 + 0.999  # in case le == 0
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2)  # b, w, c, h
        matrix_out = (torch.zeros(matrix_new.shape))
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1)  # b, c, h, w
        return matrix_out
"""