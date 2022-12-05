import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from torch.utils.data import Dataset, DataLoader
from model import *
from colorama import Fore
from metric import *
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from ray import tune
import ray
os.environ["CUDA_VISIBLE_DEVICES"]="cuda:0"

class SeedDataset(Dataset):

    def __init__(self, annotations_file):
        super().__init__()
        self.data: pd.DataFrame = pd.read_csv(annotations_file)
        self.data: pd.DataFrame = self.data[self.data['label'].notna()]
        self.Y = self.data['label']
        self.X = self.data.drop(columns=['id','label']).fillna(value=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.as_tensor(self.X.iloc[idx].values).type(torch.FloatTensor), torch.as_tensor(self.Y.iloc[idx]).type(
            torch.LongTensor)


def train(dataloader, model, loss_fn, optimizer, device, positive_weight,scheduler,schedule_mode):
    model.train()

    Y = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # print(X.shape)
        logit = model(X)
        positive_index = y == 1
        # print(logit.shape)
        # print(y.shape)
        # print(positive_index)
        # acc=Accuracy(logit.argmax(1), y)
        loss = loss_fn(logit, y)
        loss = ( (positive_weight/(1+positive_weight)) * loss_fn(logit[positive_index], y[positive_index]) + (1/(1+positive_weight))*loss_fn(logit[~positive_index], y[
            ~positive_index]) ) / len(X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if schedule_mode==1:
        #     scheduler.step()

        if batch % 50 == 0:
            loss = loss.item()
            print(f"{Fore.GREEN + '[train]===>'} loss_train: {loss} {'' + Fore.RESET}")


def valid(dataloader, model, loss_fn, device,positive_weight):
    model.eval()

    num_dataset = len(dataloader.dataset)
    loss = 0

    with torch.no_grad():
        pred, Y = [], []
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            logit = model(X)
            loss += loss_fn(logit, y).item()

            pred.append(logit.argmax(1))
            Y.append(y)

        loss /= num_dataset

        pred = torch.cat(pred)
        Y = torch.cat(Y)
        print(f"{Fore.CYAN + '[valid]===>'} " \
              f"loss_valid(train): {loss}  acc: {100 * Accuracy(pred, Y)}%  precision: {Precision(pred, Y)}  recall: {Recall(pred, Y)}   fscore: {Fscore(pred, Y,positive_weight)}" \
              f"{'' + Fore.RESET}")
    return loss,Accuracy(pred, Y),Precision(pred, Y),Recall(pred, Y),Fscore(pred, Y,positive_weight)


if __name__ == '__main__':

    torch.manual_seed(777)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("gpu")

    batch_size, in_features, out_features = 60, 25, 2
    lr, positive_weight = 1e-3, 1
    epochs = 30

    writer=SummaryWriter('./visual')


    loss_fn = nn.CrossEntropyLoss()

    


    def training_f(config):
        loss_t_list=[]
        loss_v_list=[]

        train_dataset = SeedDataset("./data/v1/train.csv")
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        valid_dataset = SeedDataset("./data/v1/valid.csv")
        valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
        
        model = FCModel( in_features, config['out_features'] )
        # if config['optim_id'] == 1:
        #     optimizer = optim.Adam( model.parameters(), lr=config['lr'] )
        # elif config['optim_id'] == 2:
        #     optimizer = optim.Adadelta( model.parameters(), lr=config['lr'] )
        # elif config['optim_id'] == 3:
        #     optimizer = optim.SGD( model.parameters(), lr=config['lr'] )
        # elif config['optim_id'] == 4:
        #     optimizer = optim.NAdam( model.parameters(), lr=config['lr'] )
        # if config['sche_id'] == 1:
        #     scheduler1 = optim.lr_scheduler.CyclicLR( optimizer,  max_lr=config['lr'],base_lr=1e-4,cycle_momentum=False)
        #     scheduler2 = optim.lr_scheduler.ReduceLROnPlateau( optimizer ,patience=100)
        #     scheduler=optim.lr_scheduler.ChainedScheduler([scheduler1,scheduler2])
        # elif config['sche_id']==2:
        #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,threshold=0.01)
        # elif config['sche_id'] == 3:
        #     scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts( optimizer,T_0=1,T_mult=2 )
        # elif config['sche_id'] == 4:
        #     scheduler = optim.lr_scheduler.StepLR( optimizer,step_size=2,gamma=0.5 )
        optimizer = optim.Adam( model.parameters(), lr=config['lr'] )
        # scheduler1 = optim.lr_scheduler.StepLR( optimizer,step_size=2,gamma=0.2)
        # scheduler2 = optim.lr_scheduler.ReduceLROnPlateau( optimizer)
        # scheduler=optim.lr_scheduler.ChainedScheduler([scheduler1,scheduler2])
        scheduler1 = optim.lr_scheduler.StepLR( optimizer,step_size=2,gamma=0.5)
        # scheduler2 = optim.lr_scheduler.ReduceLROnPlateau( optimizer)

        for t in range( config['epochs'] ):
            print( f"{Fore.GREEN + '===>'} Epoch {t + 1} {'' + Fore.RESET}\n" \
                   "---------------------------------------" )
            # if config['sche_id'] == 4:
            #     train( train_dataloader, model, loss_fn, optimizer, device, config['positive_weight'],scheduler,1 )
            # else:
            #     train( train_dataloader, model, loss_fn, optimizer, device, config['positive_weight'],scheduler,0 )
            train( train_dataloader, model, loss_fn, optimizer, device, config['positive_weight'], scheduler1, 0 )
            loss_train, _, _, _, _= valid(train_dataloader, model, loss_fn, device, config['positive_weight'])
            loss, acc, _, recall, f_score = valid( valid_dataloader, model, loss_fn, device , config['positive_weight'] )
            # if config['sche_id'] == 1:
            #     pass
            # elif config['sche_id'] == 2:
            #     scheduler.step(metrics=loss)
            # else:
            #     scheduler.step()
            scheduler1.step()
            # scheduler2.step( metrics=loss )
            torch.save( model.state_dict(), f"./checkpoints/{t}_epoc.pt" )
            tune.report( loss_train=loss_train,loss_valid=loss, acc=acc, recall=recall, f_score=f_score )
            print("loss_train={}",loss_train)
            loss_t_list.append(loss_train)
            loss_v_list.append(loss)
            _, (ax1,ax2) = plt.subplots(1,2)
            # plt.clf()
            ax1.plot(list(range( len(loss_t_list) )),loss_t_list)
            ax1.set_title('loss_train')
            ax2.plot(list(range( len(loss_v_list) )),loss_v_list)
            ax2.set_title('loss_valid')
            # plt.show()
            plt.savefig(f"lr:{config['lr']}_epochs:{config['epochs']}_sche_id:{config['sche_id']}_optim_id:{config['optim_id']}.png")



    search_space1={
        'lr':tune.grid_search([0.1,0.01,0.05,0.001,0.005,0.0001]),
        'batch_size':tune.grid_search([100,500,1000,2000,3000]),
        'epochs':tune.grid_search(list(np.array(range(5,6))*10)),
        'positive_weight': tune.grid_search([1.5]),
        'out_features': tune.grid_search( [2] ),
        'sche_id':tune.grid_search([2]),
        'optim_id': tune.grid_search( [1] ),
    }
    search_space2={
        'lr': tune.grid_search( [0.1] ),
        'epochs': tune.grid_search( list( np.array( range( 5, 6 ) ) * 10 ) ),
        'positive_weight': tune.grid_search( [3] ),
        'out_features': tune.grid_search( [2] ),
        'sche_id': tune.grid_search( [1] ),
        'optim_id': tune.grid_search( [3] )
    }
    # analysis=tune.run(
    #     training_f,
    #     config=search_space1,
    #     resources_per_trial={"cpu":8,"gpu":1},
    #     mode="max",
    #     metric="f_score",
    #     verbose=0,
    #     local_dir="D:\\ray_results"
    # )

    # torch.save(model.state_dict(), "./model/{}_ver.pt".format('1'))
    training_f({'epochs':40,'lr':0.00001,'out_features':2,'positive_weight':1,'optim_id':3,'sche_id':2,'batch_size':60})


