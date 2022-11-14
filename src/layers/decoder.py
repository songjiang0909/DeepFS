import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_dim,hidden,output_dim):
        super(MLP,self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, output_dim, bias=True)


    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x




class nonPeriodic(nn.Module):
    def __init__(self,args):
        super(nonPeriodic,self).__init__()

        self.input_len = args.input_len
        self.embed_size = args.embed_size
        self.hidden = args.embed_size
        self.out_len = args.out_len

        self.nn = MLP(input_dim=self.input_len,hidden=self.hidden,output_dim=self.out_len)
        
    def forward(self,x):
        pred = self.nn(x)
        
        return pred



class Periodic(nn.Module):

    def __init__(self,args):
        super(Periodic,self).__init__()

        self.args = args
        self.input_len = args.input_len
        self.embed_size = args.embed_size
        self.hidden = args.embed_size
        self.out_len = args.out_len
        self.base = args.base
        self.Tensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor


        self.nn_weight = MLP(input_dim=self.embed_size,hidden=self.hidden,output_dim=self.base)
        self.nn_phase = MLP(input_dim=self.embed_size,hidden=self.hidden,output_dim=self.base)


    def sine_function(self,weight,period,phase,t):
        return weight*torch.sin((self.Tensor([2*np.pi])/period)*t+phase)

    
    def generate_timestamps(self,num,length,start):

        t = [[start+i*self.args.granularity+shift*self.args.granularity for i in range(length)] for shift in range(num)]
        
        return self.Tensor(t) 

        
    def forward(self,x,timestamps):



        weights = self.nn_weight(x)
        phases = self.nn_phase(x)

        weights = torch.mean(torch.mean(weights,axis=1),axis=0)
        phases = torch.mean(torch.mean(phases,axis=1),axis=0)



        t = timestamps
        pred = 0
        start = 2
        if self.args.is_sim:
            start = 0
        for i in range(start,self.base):
            pred+= self.sine_function(weights[i],self.Tensor([i+1]),phases[i],t)


        return pred,weights,self.base,phases

        
    