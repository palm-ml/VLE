import torch.nn.functional as F 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import scipy.io as scio

from model import *
from utils import * 
from measures import * 


def softmax(d,T=1):
    for i in range(len(d)):
        d[i] = d[i]*T
        d[i] = np.exp(d[i])/sum(np.exp(d[i]))
    return d

def train(epoch, enc, dec, optimizer, data_loader, args):
    device = args['device']
    enc.train()
    dec.train()
    # records
    train_loss = []
    train_recx_loss = []
    train_recy_loss = []
    train_kl_loss = []
    # ----------------------------training----------------------------------
    for idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_data = torch.cat((batch_x, batch_y),1)
        # forward
        (mu, sigma) = enc(batch_data)
        z = mu + sigma * (torch.randn(mu.size()).to(device))
        batch_x_hat = dec(z)
        d = F.sigmoid(z[:,-args['num_class']:])
        # loss
        rec_loss_x = F.mse_loss(batch_x_hat,batch_x)
        kl_loss = gauss_kl_loss(mu,sigma)
        rec_loss_y = F.binary_cross_entropy(d,batch_y)
        loss = rec_loss_y + args['alpha'] * kl_loss + args['beta'] * rec_loss_x
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record
        train_loss.append(loss.data.cpu())
        train_recx_loss.append(rec_loss_x.data.cpu())
        train_kl_loss.append(kl_loss.data.cpu())
        train_recy_loss.append(rec_loss_y.data.cpu())
    # print
    if (epoch + 1) % 10 == 0:
        print('Epoch {:04d}: '.format(epoch + 1))
        print('loss: {:.03f} '.format(np.mean(train_loss)), 
              'kl_loss: {:.03f} '.format(np.mean(train_kl_loss)),
              'recx_loss: {:.03f} '.format(np.mean(train_recx_loss)),
              'recy_loss: {:.03f}'.format(np.mean(train_recy_loss)))

def label_enhance(enc, datas, labels, args):
    device = args['device']
    enc.eval()
    n_samples = len(datas)
    indices = np.arange(n_samples)
    n_batches = n_samples // args['batch_size']
    if n_batches * args['batch_size'] < n_samples:
        n_batches += 1

    distributions = []
    for i in range(n_batches):
        offset = i*args['batch_size']
        if offset + args['batch_size'] > n_samples:
            cur_indices = indices[offset:]
        else:
            cur_indices = indices[offset:offset+args['batch_size']]
        batch_x = datas[cur_indices].to(device)
        batch_y = labels[cur_indices].to(device)
        batch_data = torch.cat((batch_x, batch_y),1)
        # forward
        (mu, sigma) = enc(batch_data)
        d = F.sigmoid(mu[:,-args['num_class']:])
        distributions.extend(d.data.cpu().numpy())
    return distributions

def save_data(distribution, args):
    if not p.isdir(args['dst_path']):
        os.mkdir(args['dst_path'])
    data_folder = p.join(args['dst_path'], args['dataset'])
    if not p.isdir(data_folder):
        os.mkdir(data_folder)
    dst_path = p.join(data_folder,args['dataset'])+'_LE.mat'
    distribution = np.array(distribution, dtype = np.float64)

    mat_data ={ 'distributions':distribution}
    scio.savemat(dst_path, mat_data)

def main(args):
    device = torch.device('cuda:'+str(args['gpu']) if torch.cuda.is_available() else 'cpu')
    args['device'] = device
    setup_seed(args['seed'])
    # create data
    train_data = MLLDataset(args)
    train_loader = DataLoader(train_data, batch_size=args['batch_size'],shuffle=True)
    args['num_class'] = train_data.n_label
    # create model
    enc = VAE_Encoder(n_in=train_data.n_feature + train_data.n_label, n_hidden=args['n_hidden'], n_out=args['dim_z'], keep_prob=args['keep_prob'])
    dec = VAE_Bernulli_Decoder(n_in=args['dim_z'], n_hidden=args['n_hidden'], n_out=train_data.n_feature, keep_prob=args['keep_prob'])
    optimizer = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),lr=args['learning_rate'],weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[500,800,900], gamma=0.2, last_epoch=-1)
    # training
    print('Begin training pharse')
    for epoch in range(args['epochs']):
        scheduler.step()
        train(epoch,enc,dec,optimizer,train_loader,args)
    # enhance label
    distribution = label_enhance(enc,train_data.features, train_data.labels, args)
    # save distributions
    save_data(distribution, args)

    """
    Test the label enhancement results
    """
    # load tragets 
    data_folder = p.join(args['src_path'],args['dataset'])
    file_path = p.join(data_folder,args['dataset'])+'_d.plk'
    with open(file_path,'rb') as f:
        targets = pickle.load(f)
    preds = softmax(np.array(distribution))

    
    dists = []
    dist1 = chebyshev(targets, preds)
    dist2 = clark(targets, preds)
    dist3 = canberra(targets, preds)
    dist4 = kl_dist(targets, preds)
    dist5 = cosine(targets, preds)
    dist6 = intersection(targets, preds)

    dists.append(dist1)
    dists.append(dist2)
    dists.append(dist3)
    dists.append(dist4)
    dists.append(dist5)
    dists.append(dist6)
    print(np.round(dists,3))

