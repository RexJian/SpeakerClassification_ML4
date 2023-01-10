import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self,device,d_model=512,n_spks=600,dropout=0.1):
        super(Model, self).__init__()
        self.present=nn.Linear(40,d_model)

        self.encoder_layer=nn.TransformerEncoderLayer(d_model=d_model,dim_feedforward=256,nhead=1)

        self.pred_layer=nn.Sequential(
            nn.Linear(d_model,n_spks),
            # nn.ReLU(),
            # nn.Linear(d_model,n_spks)
        )
        self.loss=nn.CrossEntropyLoss()
        self.loss.to(device)
    def forward(self,mels):
        out=self.present(mels)
        out=out.permute(1,0,2)
        out=self.encoder_layer(out)
        out=out.transpose(0,1)
        stats=out.mean(dim=1)
        out=self.pred_layer(stats)
        return out
    def cal_loss_accuracy(self,outputs,targets,batch_size):
        loss=self.loss.forward(outputs,targets)
        accuracy=(outputs.argmax(1)==targets).sum()/batch_size
        return loss,accuracy
