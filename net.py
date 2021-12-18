import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.layer1=nn.Linear(1,1024)
        self.layer2=nn.Linear(1024,32)
        self.layer3=nn.Linear(32,4)
        self.layer4=nn.Linear(4,1)
    def forward(self,x):
        x=self.layer1(x)
        x=F.sigmoid(x)
        x=self.layer2(x)
        x=F.sigmoid(x)
        x=self.layer3(x)
        x=F.sigmoid(x)
        x=self.layer4(x)
        x=F.relu(x)
        return x
        
my_nn=Net().to("cuda")
count=0
acc=0
for k in range(1,11):
    for i in range(1,2000):
        my_optim=optim.SGD(my_nn.parameters(),lr=0.0005/k,momentum=0.9,weight_decay=0.1)
        my_optim.zero_grad()
        j=torch.rand(1)
        act=torch.tensor([j[0]**3],dtype=torch.float,device="cuda")
        output=my_nn(torch.tensor([j[0]],dtype=torch.float,device="cuda"))
        loss=F.mse_loss(act,output)
        acc+=loss
        count+=1
        loss.backward()
        my_optim.step()
    print(loss/count)
