
import numpy as np 
import os.path as p 
import pickle
import scipy.io as scio
from measures import *

PATH = p.dirname(__file__)

def softmax(d,T=1):
    for i in range(len(d)):
        d[i] = d[i]*T
        d[i] = np.exp(d[i])/sum(np.exp(d[i]))
    return d

def linear_norm(d):
    for i in range(len(d)):
        d[i] = d[i] / sum(d[i])
    return d

def load_data(src_path1,src_path2,dataset):
    src_file1 = p.join(src_path1,dataset,dataset+'_d'+'.plk')
    src_file2 = p.join(src_path2,dataset,dataset+'_LE.mat')
    with open(src_file1,'rb') as f:
        target_train = pickle.load(f)
    target_train = softmax(target_train)
    datas = scio.loadmat(src_file2)
    preds_train = datas['distributions']
    preds_train = softmax(preds_train)

    targets = target_train
    preds = preds_train
    return targets, preds
 
def save_data(preds, targets, dst_path):
    scio.savemat(dst_path,{'LE_res':preds,'LD':targets})

def calcaulate_dists(preds, targets):
    dists = [] 
    dist1 = chebyshev(targets, preds)
    dist2 = clark(targets, preds)
    dist3 = canberra(targets, preds)
    dist4 = kl_dist(targets, preds)
    dist5 = cosine(targets, preds)
    dist6 = intersection(targets, preds)
    dists.append(dist1)
    dists.append(dist5)
    dists.append(dist2)
    dists.append(dist3)
    dists.append(dist4)
    dists.append(dist6)
    return dists

if __name__ =='__main__':
    datasets = ['Yeast_alpha','Yeast_cdc','Yeast_cold','Yeast_diau','Yeast_dtt','Yeast_elu',
'Yeast_heat','Yeast_spo','Yeast_spo5','Yeast_spoem','SJAFFE','SBU_3DFE','Movie']

    dataset = datasets[12]
    src_path1 = p.join(PATH,'Datasets','datas')
    src_path2 = p.join(PATH,'distribution_zry')

    targets, preds = load_data(src_path1,src_path2,dataset)
    dists = calcaulate_dists(preds, targets)
    dists = np.array(dists)
    print(dataset)
    print(np.round(dists,3))
    with open('dst','w') as f:
        for d in dists:
            f.write(str(np.round(d,3)))
            f.write('\t')
    # dst_path = p.join(PATH,'distribution_zry',dataset,dataset+'_VLE.mat')
    # save_data(preds, targets, dst_path)
    # a = np.array([0.1,0.2,0.3,0.4])
    # a = a[None,:]
    # print(softmax(a))

