import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class BidirectionalLSTM_skip(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM_skip, self).__init__()
        self.hidden_size=hidden_size
        self.d1=self.make_layer(input_size,output_size,26)
        self.exit = nn.Linear(hidden_size, output_size)
    def make_layer(self,input_size,hidden_size,t):
        layers=[]
        for i in range(0, t):
            if i == 0:
                layers.append(skip_block(input_size,hidden_size,shortcut=False))
            else:
                layers.append(skip_block(input_size,hidden_size,shortcut=True))
        return layers
    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        batch_size=input.size(0)
        num_length=input.size(1)
        chunk_split=input.chunk(num_length,dim=1)
        h,c = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))
        out_puts=[]
        for chunk,layer in zip(chunk_split,self.d1):
            chunk=chunk.squeeze(1)
            h,c=layer(chunk,h,c)
            out_puts.append(h.unsqueeze(1))
        output=torch.cat(out_puts,dim=1)
        output=self.exit(output)
        return output


class skip_block(nn.Module):
    def __init__(self,input_size, hidden_size,shortcut=True):
        super(skip_block, self).__init__()
        self.block1=nn.LSTMCell(input_size, hidden_size).to(device)
        self.shortcut=shortcut
    def forward(self, input,prehidden_h,prehidden_c):
        h,c=self.block1(input,(prehidden_h,prehidden_c))

        if self.shortcut:
            h+=prehidden_h
            c+=prehidden_c
        else:
            pass
        return h,c

if __name__ == '__main__':
    print('hello')
    ass=torch.randn(100,26,512)
    ass=ass.to(device)
    cell0=BidirectionalLSTM_skip(512,256,256).to(device)
    b=cell0(ass)
    print(b.shape)