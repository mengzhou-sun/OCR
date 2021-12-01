
from torch import nn
from torch.autograd import Variable
print_shape_flag = False
import torch.onnx


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_max_len, vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_max_len = output_max_len
        self.vocab_size = vocab_size
        # self.emmbedding=nn.Linear(in_features=72072,out_features=vocab_size)
    def forward(self, src, tar, src_len,  train=True):
        tar = tar.permute(1, 0)
        batch_size = src.size(0)
        # max_len = tar.size(0) # <go> true_value <end>

        #outputs = Variable(torch.zeros(self.output_max_len-1, batch_size, self.vocab_size))
        out_enc = self.encoder(src)
        output=self.decoder(out_enc)

        # for t in range(0, self.output_max_len - 1):  # max_len: groundtruth + <END>
        #    output= self.decoder(out_enc)
        #    print(output.shape)
           #T,b,h=output.size()
           # print(output.size())#([1, 693, 104]))
           #t_rec = output.view(T , b* h)

           #output_l = self.emmbedding(t_rec)



           # outputs[t] = output

           #output = Variable(self.one_hot(tar[t + 1].data))
        #
        #     attns.append(attn_weights.data.cpu())  # [(32, 55), ...]
        return output

    def one_hot(self, src):  # src: torch.cuda.LongTensor
        ones = torch.eye(self.vocab_size)
        return ones.index_select(0, src)