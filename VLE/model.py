import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 


class VAE_Encoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Encoder,self).__init__()
        self.n_out = n_out
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hidden,n_out*2)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
    
    def forward(self,inputs,eps=1e-8):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        out = self.fc_out(h1)
        mean = out[:,:self.n_out]
        std = F.softplus(out[:,self.n_out:]) + eps
        return (mean,std)

class VAE_Bernulli_Decoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Bernulli_Decoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.fc_out = nn.Linear(n_hidden,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        out = F.sigmoid(self.fc_out(h1))
        return out

class VAE_Gauss_Decoder(nn.Module):
    def __init__(self,n_in,n_hidden,n_out,keep_prob=1.0):
        super(VAE_Gauss_Decoder,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(n_in,n_hidden),
                                    nn.Tanh(),
                                    nn.Dropout(1-keep_prob))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden,n_hidden),
                                    nn.ELU(),
                                    nn.Dropout(1-keep_prob))
        self.fc_mean = nn.Linear(n_hidden,n_out)
        self.fc_var = nn.Linear(n_hidden,n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self,inputs):
        h0 = self.layer1(inputs)
        h1 = self.layer2(h0)
        mean = self.fc_mean(h1)
        var = F.softplus(self.fc_var(h1))
        return mean,var
