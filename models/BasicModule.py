#coding:utf8
import torch
import time
import os


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(self)[:-4]

    def save(self, ckp_dir):
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)

        filename = os.path.join(ckp_dir, self.model_name+'.pth')
        torch.save(self.state_dict(),filename)


    def load(self, ckp_dir):
        if not torch.cuda.is_available():
            state = torch.load(os.path.join(ckp_dir, self.model_name+'.pth'),
                                 map_location=lambda storage, loc: storage)
        else:
            state = torch.load(os.path.join(ckp_dir, self.model_name+'.pth'))
        self.load_state_dict(state)
