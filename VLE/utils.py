import os 
import os.path as p 
import numpy as np 
import torch
import pickle
import random
from torch.utils.data import Dataset




def setup_seed(seed):
    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loadData(args):
    data_folder = p.join(args['src_path'],args['dataset'])
    file_path = p.join(data_folder,args['dataset'])+'.plk'
    # 载入数据
    with open(file_path,'rb') as f:
        info = pickle.load(f)
        data = pickle.load(f)
    return info,data


def mll_rec_loss(preds, targets, eps = 1e-12):
    w1 = 1 / targets.sum(1)
    loss = -targets*(torch.log(preds + eps))
    loss = loss.sum(1)*w1
    return loss.mean(0)


def gauss_kl_loss(mu,sigma,eps = 1e-12):
    mu_square = torch.pow(mu,2)
    sigma_square = torch.pow(sigma,2)
    loss = mu_square + sigma_square - torch.log(eps+sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()


def t_softmax(d,t=1):
    for i in range(len(d)):
        d[i] = d[i]*t
        d[i] = np.exp(d[i])/sum(np.exp(d[i]))
    return d

class MLLDataset(Dataset):
    def __init__(self, args):
        super(MLLDataset, self).__init__()
        info, datas = loadData(args)
        self.n_feature = info['n_feature']
        self.n_label = info['n_label']
        self.sparse = info['sparse']
        self.genDataSets(datas)

    def genDataSets(self,datas):
        dataSet = []
        n_sample = datas['length']
        feature_data = datas['data']
        label_data = datas['label']
        features = torch.zeros(n_sample,self.n_feature)
        labels = torch.zeros(n_sample,self.n_label)
        if self.sparse:
            for i in range(n_sample):
                feature = torch.from_numpy(np.array(feature_data[i],dtype=np.int64))
                label = torch.from_numpy(np.array(label_data[i],dtype=np.int64))
                if len(feature) > 0:
                    features[i].scatter_(0, feature,1)
                if len(label) > 0:
                    labels[i].scatter_(0,label,1)
        else:
            for i in range(n_sample):
                feature = torch.from_numpy(np.array(feature_data[i]))
                label = torch.from_numpy(np.array(label_data[i]))
                features[i] = feature
                labels[i] = label
            # Normalization
            max_data = features.max()
            min_data = features.min()
            features = (features-min_data) / (max_data - min_data)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index],self.labels[index]