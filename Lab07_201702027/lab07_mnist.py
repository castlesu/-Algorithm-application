import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

def File_Open(train,test):
     train= pd.read_csv(train)
     x_train = train.drop("label", axis=1).to_numpy() / 255
     y_train = train['label']
     y_list =np.zeros(shape=(y_train.size,10))
     for i, y in enumerate(y_train):
        y_list[i][y] = 1
     y_train = y_list
     test = pd.read_csv(test)
     x_test = test.to_numpy() / 255

     return x_train, y_train ,x_test

def Get_Acc(pred, answer):
    correct = 0
    for p, a in zip(pred,answer):
        pv, pi =p.max(0)  #
        av, ai =a.max(0)
        if pi == ai:
            correct += 1

    return  correct/len(pred)

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel,self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 264)
        self.fc3 = nn.Linear(264, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self,x):
        x = torch.relu(self.fc1(x)) #relu는 양수로 처리
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x

def train(x_train,y_train,batch, lr, epoch): # LR:learning rate
    model = MNISTModel()
    model.train()

    loss_function = nn.MSELoss(reduction="mean") #틀린거 확인
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #data 처리
    x=torch.from_numpy(x_train).float()
    y=torch.from_numpy(y_train).float()

    data_loader = torch.utils.data.DataLoader(list(zip(x,y)),batch,shuffle=True)

    epoch_loss =[]
    epoch_acc =[]
    for e in range(epoch):
        total_loss = 0
        total_acc =0
        for data in data_loader:
            x_data,y_data = data

            pred = model(x_data) #forward(문제풀이)
            loss = loss_function(pred,y_data) #채점,학습

            optimizer.zero_grad() #이전학습 리셋
            loss.backward() #학습 진행
            optimizer.step() #학습결과 업데이트(학습반영)

            total_loss += loss.item()
            total_acc += Get_Acc(pred,y_data)

        epoch_loss.append(total_loss/len(data_loader))
        epoch_acc.append(total_acc/len(data_loader))

    return model , epoch_loss,epoch_acc

def test(model,x_test,batch):
    model.eval() #평가모드
    x = torch.from_numpy(x_test).float()
    data_loader = torch.utils.data.DataLoader(x, batch ,shuffle=False)

    preds=[]
    for data in data_loader:
        pred = model(data)
        for p in pred:
            pv, pi = p.max(0)
            preds.append(pi.item())

    return preds

def Save_Pred(pred_path, preds):
    df = pd.DataFrame(preds)
    df.rename(columns={0:'Sentiment'}, inplace=True)
    df.index +=1
    df.to_csv(pred_path,index=True,index_label='ImageId')

def Draw_Graph(loss,acc):
    loss,acc
    loss = np.array(loss)
    acc = np.array(acc)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(loss.T)
    ax1.set_xlabel('Epoch per Loss')
    ax2.plot(acc.T)
    ax2.set_xlabel('Epoch per Accuracy')

    plt.show()


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    save_path = 'data/my_submission.csv'

    batch =128
    learning_rate = 0.001
    epoch =10

    x_train,y_train,x_test= File_Open(train_path,test_path)
    model,epoch_loss,epoch_acc = train(x_train,y_train,batch,learning_rate,epoch)
    preds = test(model,x_test,batch)

    Save_Pred(save_path,preds)
    Draw_Graph(epoch_loss,epoch_acc)
