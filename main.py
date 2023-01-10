import math
import json
import random
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

from MyDataset import *
from Model import *

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def TrainOptimize(train_dataloader,valid_dataloader):
    model=Model(device=DEVICE)
    model=model.to(DEVICE)
    pre_total_loss=9999999
    optim=torch.optim.AdamW(model.parameters(),HyperParam['learn_rate'])

    warmup_steps=HyperParam['warnup_steps']
    T_max=HyperParam['epoches']
    lr_max=HyperParam['learn_rate']
    lr_min=lr_max/100
    total_steps=HyperParam['epoches']*len(train_dataloader.dataset)/HyperParam['batch_size']
    lambda1=lambda cur_iter:cur_iter/warmup_steps if cur_iter<warmup_steps \
                                                  else (0.5*(1+math.cos(math.pi*0.5*2*((cur_iter-warmup_steps)/(total_steps-warmup_steps)))))

    # lambda1=lambda cur_iter:cur_iter/warmup_steps if cur_iter<warmup_steps \
    #                                             else (lr_min+0.5*(lr_max-lr_min)*(1+math.cos((cur_iter-warmup_steps)/(T_max-warmup_steps)*math.pi)))/0.1
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,lambda1)

    for epoch_cnt,epoch in enumerate(range(HyperParam['epoches'])):
        torch.cuda.empty_cache()
        model.train()
        total_train_loss=0.0
        total_valid_loss=0.0
        total_accuracy=0.0
        for data in train_dataloader:
            inputs,targets=data
            inputs,targets=inputs.to(DEVICE),targets.to(DEVICE)
            outputs=model(inputs)
            loss,accuracy=model.cal_loss_accuracy(outputs,targets,HyperParam['batch_size'])
            total_train_loss+=loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
        print("Valid")
        model.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                inputs,targets=data
                inputs,targets=inputs.to(DEVICE),targets.to(DEVICE)
                outputs=model(inputs)
                loss,accuracy=model.cal_loss_accuracy(outputs,targets,HyperParam['batch_size'])
                total_valid_loss+=loss.item()
                total_accuracy+=accuracy.item()
        if epoch_cnt is 1 or pre_total_loss>total_train_loss+total_valid_loss:
            pre_total_loss=total_train_loss+total_valid_loss
            torch.save(model.state_dict(),"C:/Users/Jian/Desktop/SpeakerClassify.pth")
            print("*{}".format(epoch_cnt)+
                  "TotalTrainLoss: {}".format(round(total_train_loss,2))
                  +"  TotalValidlOSS: {}.".format(round(total_valid_loss,2))
                  +"  TotalAccuracy: {}".format(round(total_accuracy/ValidDataLoader.__len__(),2)))
        else:
            print("{}".format(epoch_cnt) +
                  "TotalTrainLoss: {}".format(round(total_train_loss, 2))
                  + "  TotalValidlOSS: {}.".format(round(total_valid_loss, 2))
                  + "  TotalAccuracy: {}".format(round(total_accuracy/ValidDataLoader.__len__(),2)))
def Test(model_name):
    model=Model(DEVICE)
    model.load_state_dict(torch.load(model_name))
    pred_list=[]
    for data in TestDataLoader:
        preds=model(data)
        for pred in preds.argmax(1).numpy().tolist():
            pred_list.append(MappingData['id2speaker'][str(pred)])
    with open('predict.csv',"w") as f:
        f.write("Id,Category\n")
        for id,category in zip(TestUtterList,pred_list):
            f.write(f"{id},{category}\n")



def TrainCollateBatch(batch):
    mel,speaker=zip(*batch)
    mel=pad_sequence(mel,batch_first=True,padding_value=-20)
    return mel,torch.LongTensor(speaker)

def TestCollateBatch(batch):
    mel = pad_sequence(batch, batch_first=True, padding_value=-20)
    return mel
HyperParam={
    'train_ratio':0.9,
    'segment_len':256,
    'batch_size':128,
    'epoches':100,
    'learn_rate':0.001,
    'warnup_steps':1000
}
DataDir='./ml2021spring-hw4'
TrainPth=Path(DataDir)/"metadata.json"
MappingPth=Path(DataDir)/"mapping.json"
TestPth=Path(DataDir)/"testdata.json"

MetaData=json.load(TrainPth.open())['speakers']
MappingData=json.load(MappingPth.open())
TestData=json.load(TestPth.open())
UtterList=[]
SpeakerList=[]
TestUtterList=[]
for speaker in MetaData:
    for utter in MetaData[speaker]:
        UtterList.append(utter['feature_path'])
        SpeakerList.append(MappingData['speaker2id'][speaker])
TrainX,ValidX,TrainY,ValidY=train_test_split(UtterList,SpeakerList,train_size=HyperParam['train_ratio'],random_state=20)

for utter in TestData['utterances']:
    TestUtterList.append(utter['feature_path'])

TrainDataset=MyDataset(DataDir,HyperParam['segment_len'],TrainX,TrainY)
ValidDataset=MyDataset(DataDir,HyperParam['segment_len'],ValidX,ValidY)
TestDataset=MyDataset(DataDir,HyperParam['segment_len'],TestUtterList)

TrainDataLoader=DataLoader(TrainDataset, HyperParam['batch_size'], shuffle=True, collate_fn=TrainCollateBatch, )
ValidDataLoader=DataLoader(ValidDataset, HyperParam['batch_size'], shuffle=True, collate_fn=TrainCollateBatch, )
TestDataLoader=DataLoader(TestDataset, HyperParam['batch_size'], shuffle=False, collate_fn=TestCollateBatch, )
TrainOptimize(TrainDataLoader,ValidDataLoader)
Test("C:/Users/Jian/Desktop/SpeakerClassify.pth")