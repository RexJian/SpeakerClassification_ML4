import random

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

class MyDataset(Dataset):

    def __init__(self,data_dir,segment_len,x, y=None):
        if y is not None:
            self.y=torch.LongTensor(y)
        else:
            self.y=y

        self.x=x
        self.data_dir=data_dir
        self.segment_len=segment_len
    def __getitem__(self, index):

        # mel = torch.load(self.data_dir + "/Dataset/{}".format(self.x[index]))
        mel=torch.load("C:/Users/Jian/Desktop/Dataset/{}".format(self.x[index]))
        if len(mel)>self.segment_len:
            start=random.randint(0,len(mel)-self.segment_len)
            mel=torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel=torch.FloatTensor(mel)
            # dim_difference=self.segment_len-len(mel)
            # pad=nn.ZeroPad2d((0,0,0,dim_difference))
            # mel=pad(mel)



        if self.y is not None:
            return mel,self.y[index]
        else:
            return mel

    def __len__(self):
        return len(self.x)
