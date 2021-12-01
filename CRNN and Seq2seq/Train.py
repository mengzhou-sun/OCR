import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import torch.utils.data
from newDecode import GCRNNDecoder
from Encode import GCRNNEncoder
from load import get_train_dataset,get_validation_dataset
from torch.autograd import Variable
from seq2seq import Seq2Seq


batch_size=1
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
vocab_size=104
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

def sort_batch(batch):
    global img, label, img_width


    n_batch = len(batch)
    input_lengths = torch.zeros(len(batch, ), dtype=torch.int)
    train_in = []
    train_in_len = []
    train_out = []
    for i in range(n_batch):
        img, label,img_width = batch[i]
        label = list(map(int, label))
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)


    train_in=np.array(train_in,dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')
    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)
    train_in_len, idx = train_in_len.sort(dim=0, descending=True)
    input_lengths[idx]=output_max_len-1
    train_out = train_out[idx]
    train_in=train_in[idx]
    return  train_in,train_out,train_in_len,input_lengths

train_loader = torch.utils.data.DataLoader(get_train_dataset(), collate_fn=sort_batch,batch_size=batch_size,shuffle=True
                                      )


valid_loader = torch.utils.data.DataLoader(get_validation_dataset(),  collate_fn=sort_batch,batch_size=batch_size
                                            )
"""
class LabelSmoothing(torch.nn.Module):

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

log_softmax = torch.nn.LogSoftmax(dim=-1)
"""
def loss_label_smoothing(predict, gt):
    def smoothlabel_torch(x, amount=0.25, variance=5):
        mu = amount/x.shape[0]
        sigma = mu/variance
        noise = np.random.normal(mu, sigma, x.shape).astype('float32')
        smoothed = x*torch.from_numpy(1-noise.sum(1)).view(-1, 1) + torch.from_numpy(noise)
        return smoothed

    def one_hot(src): # src: torch.cuda.LongTensor
        ones = torch.eye(vocab_size).to(device)
        return ones.index_select(0, src)

    gt_local = one_hot(gt.data)
    gt_local = smoothlabel_torch(gt_local)
    loss_f = torch.nn.BCEWithLogitsLoss()
    res_loss = loss_f(predict, gt_local)
    return res_loss
#log_softmax = torch.nn.LogSoftmax(dim=-1)
#crit = LabelSmoothing(vocab_size, tokens['PAD_TOKEN'], 0.4)

def train(train_loader, seq2seq, opt,  epoch):
    global num
    seq2seq.train()
    total_loss = 0
    for num,( train_in,train_out,train_in_len,input_length) in enumerate(train_loader):
        train_in=train_in.squeeze(dim=1)
        train_out=train_out.squeeze(dim=1)
        train_in = train_in.permute(0, 3, 1, 2)
        train_in, train_out = train_in.to(device),train_out.to(device)


        output= seq2seq(train_in, train_out, train_in_len)




        # print('output.shape', output.shape)
        # print('train_out.shape', train_out.shape)
        # print('input_length', input_length)
        # print('train_in_len', train_in_len)
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        loss=ctc_loss(output,train_out,input_length,train_in_len)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.data

    total_loss /= (num + 1)
    return total_loss
def valid(valid_loader,seq2seq,epoch):
    global num, test_in, test_out
    seq2seq.train()
    total_loss = 0
    for num,( train_in,train_out,train_in_len,input_length) in enumerate(valid_loader):
        train_in = train_in.squeeze(dim=1)
        train_out = train_out.squeeze(dim=1)

        train_in = train_in.permute(0, 3, 1, 2)
        train_in, train_out = train_in.to(device), train_out.to(device)
        output = seq2seq(train_in, train_out, train_in_len)
        ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        # loss = F.cross_entropy(output_l,train_label,ignore_index=tokens['PAD_TOKEN'])
        loss = ctc_loss(output, train_out, input_length, train_in_len)

        #loss = F.cross_entropy(output_l.view(-1, vocab_size),train_label,ignore_index=tokens['PAD_TOKEN'])

        total_loss += loss.data

    total_loss /= (num + 1)
    return total_loss

HIDDEN_SIZE_ENC=512
HEIGHT=64
WIDTH=16
HIDDEN_SIZE_DEC = 128
Bi_GRU=True
epochs=20
#EMBEDDING_SIZE=60
output_max_len=100
lr_milestone = [20, 40, 60, 80, 100]
global epoch
#encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU)
#decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention,tradeoff_context_embed=None)
encoder=GCRNNEncoder().to(device)
decoder=GCRNNDecoder(vocab_size,HIDDEN_SIZE_DEC,output_max_len).to(device)
seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).to(device)
opt = optim.Adam(seq2seq.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(opt, lr_milestone, gamma=0.5)

for epoch in range(0, epochs):
        scheduler.step(epoch)
        lr = scheduler.get_lr()

        loss = train(train_loader, seq2seq, opt, epoch)

        print('epoch %d/%d, loss=%.3f, ' %(epoch, epochs, loss ))
        loss_v = valid(valid_loader, seq2seq, epoch)
        print('  Valid loss=%.3f' % (loss_v ))
torch.save(seq2seq.state_dict(),'model4')
"""
valid_loader1 = torch.utils.data.DataLoader(get_train_dataset(), batch_size=4,collate_fn = sort_batch )
#torch.save(seq2seq.state_dict(),'model1')
seq2seq.load_state_dict(torch.load('model4'))
model=seq2seq.eval()
model.eval()
for num,( train_in,train_out,train_in_len) in enumerate(valid_loader1):
    train_in = train_in.squeeze(dim=1)
    train_out = train_out.squeeze(dim=1)
    
    out=model(train_in,train_out,train_in_len)
    print('0000',train_out[0])
    a=out[0].permute(1,0,2)
    print(a[0].argmax(dim=0))

"""

