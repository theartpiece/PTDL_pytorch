import torch
from torch.utils.data import DataLoader
import  argparse
import csv,numpy as np
import pandas as pd
parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
import datasets
import models
import utils
import os
from early_stopping import  EarlyStopping
from torch.utils.tensorboard import SummaryWriter
# general settings
parser.add_argument('--dataset', default=None, help='network name')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--epochs', default=5000,type=int)
parser.add_argument('--cuda_device',type=int,default=0,
                    help='the cuda device')
parser.add_argument('--lr', type=float,default=1e-3)
parser.add_argument('--num_workers', type=int,default=0, help="for dataloader, todo: make appropriate changes if reqd")
parser.add_argument('--seed', type=int,default=1000, help="random seed for splits generation")
parser.add_argument('--save_to', type=str,default=None, help="Where to save the model")
parser.add_argument('--load_model', type=int,default=0, help="Whether to load model or not?")
parser.add_argument('--log', type=str,default=None, help="name of the log writer")

# ablation settings
parser.add_argument('--skip_proto_layer', type=int,default=0, help="skip the proto layer")
parser.add_argument('--skip_decode', type=int,default=0, help="skip the decode layer")

args=parser.parse_args()
writer = SummaryWriter(f'runs/{args.log }')
train_data_file=f"./Data/{args.dataset}_train.csv"
test_data_file=f"./Data/{args.dataset}_test.csv"
val_data_file=f"./Data/{args.dataset}_val.csv"
model_savefile=f"./Models/{args.save_to}" if args.save_to is not None else f"./Models/{args.dataset}"
train_data = pd.read_csv(train_data_file, delimiter=",")
test_data = pd.read_csv(test_data_file, delimiter=",")
train_data = train_data.sample(frac=1.0, replace=False, random_state=1000).reset_index(drop=True)
if os.path.exists(val_data_file):
    # print("here1")
    val_data = pd.read_csv(val_data_file, delimiter=",")
else:
    # print("here2")
    val_data = train_data.iloc[:5000]
    train_data = train_data.iloc[5000:]
    val_data.to_csv(val_data_file,index=False)
    train_data.to_csv(train_data_file,index=False)
val_data= torch.tensor(val_data.values)
train_data = torch.tensor(train_data.values)

test_data = torch.tensor(test_data.values)

mnist_train_dataset= datasets.MNISTDataset(train_data[:, 0], train_data[:, 1:].view(-1,1,28,28).float())
mnist_val_dataset= datasets.MNISTDataset(val_data[:, 0], val_data[:, 1:].view(-1,1,28,28).float())
mnist_test_dataset= datasets.MNISTDataset(test_data[:, 0], test_data[:, 1:].view(-1,1,28,28).float())

mnist_train_dler=DataLoader(mnist_train_dataset,batch_size=args.batch_size,shuffle=True)
mnist_val_dler=DataLoader(mnist_val_dataset,batch_size=args.batch_size,shuffle=False)
mnist_test_dler=DataLoader(mnist_test_dataset,batch_size=args.batch_size,shuffle=False)

model=models.ProtoTypeDL(args,k=10).cuda()
if args.load_model==1:
    model.load_state_dict(torch.load(model_savefile))
test_loss, test_acc = utils.evaluate(mnist_test_dler, model)
print('\033[95m INITIAL TEST-loss %.5f TEST-Acc %.5f \033[0m' % (test_loss, test_acc))

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

es=EarlyStopping(-np.inf,patience=10)

model.train()
utils.ElasticDeformation.initialize()
for epoch in range(args.epochs):
    if epoch%10==0: print(f"epoch {epoch}")
    train_total_loss = 0
    train_total_reco_loss = 0

    for train_batch in mnist_train_dler:
        batch,y=train_batch[0].cuda(),train_batch[1].cuda()
        batch=utils.ElasticDeformation.getElasticDeformation(batch)
        loss=model.loss(batch,y)
        train_total_loss+=len(train_batch)*loss.item()
        train_total_reco_loss+=len(train_batch)*model.ae_loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar("Loss/Train/Total_loss",train_total_loss/50000,epoch)
    writer.add_scalar("Loss/Train/Total_Reconstruction_loss",train_total_reco_loss/50000,epoch)
    # if (epoch+1)%3==0:
    eval_loss,eval_acc=utils.evaluate(mnist_val_dler,model)
    writer.add_scalar("Loss/Validation/total_loss",eval_loss,epoch)
    writer.add_scalar("Accuracy/Validation",eval_acc,epoch)
    es(-eval_loss,epoch,model)
    if es.early_stop:
        break
    if es.improved:
        if epoch>=100:
            torch.save(model.state_dict(), model_savefile)
        test_loss, test_acc = utils.evaluate(mnist_test_dler, model)
        print('\033[95m TEST-loss %.5f TEST-Acc %.5f \033[0m' % (test_loss,test_acc))
        writer.add_scalar("Loss/Test/Total_loss",test_loss,epoch)
        writer.add_scalar("Accuracy/Test",test_acc,epoch)




