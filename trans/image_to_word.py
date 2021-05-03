import torch
import torch.nn as nn

class im_convert_word(nn.Module):
    def __init__(self,input_size,output_size):
        super(im_convert_word, self).__init__()
        self.liner=nn.Linear(input_size,output_size)
        self.rulu=nn.ReLU()
    def forward(self, input):
        batch_size=input.size(0)
        output=input.view(batch_size,-1)
        output=self.liner(output)
        output=self.rulu(output)
        return output
if __name__ == '__main__':
    tensors=torch.randn(100,26,512)
    big_size=tensors.view(100,-1).size(-1)
    a=im_convert_word(big_size,38)
    b=a(tensors)
    print(b)
    print(b.shape)
