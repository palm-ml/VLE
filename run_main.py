import argparse
import os.path as p 
from main import main

PATH = p.dirname(__file__)

dataset_list = ['SBU_3DFE','SJAFFE']

parser = argparse.ArgumentParser(description='VAE_LE  process')

parser.add_argument('--dataset','-d',type=str, default=dataset_list[1])

# training args
parser.add_argument('--epochs', '-e', type=int, default=150,
                    help = 'number of epochs to train (default: 500)')
parser.add_argument('--learning_rate','-lr', type=float, default=0.001,
                    help = 'learning rate (default: 0.001)')
parser.add_argument('--batch_size','-b', type=int, default=128,
                    help = 'bacth size of the training set (default: 128)')
parser.add_argument('--keep_prob','-k', type=float, default=0.9,
                    help = 'keep ratio of the dropout settings (default: 0.9)')

# model args
parser.add_argument('--n_hidden','-hidden', type=int, default=150,
                    help = 'number of the hidden nodes (default: 150)')
parser.add_argument('--dim_z','-dim_z', type=int, default=50,
                    help='dimension of the variable Z (default: 100)')
parser.add_argument('--alpha','-a', type=float, default=1.0, 
                    help = 'balance parameter of the loss function (default=1.0)')
parser.add_argument('--beta','-beta', type=float, default=1.0, 
                    help = 'balance parameter of the loss function (default=1.0)')

# other args
parser.add_argument('--gpu', '-gpu', type = int, default = 0, 
                    help = 'device of gpu id (default: 0)')
parser.add_argument('--seed', '-seed', type = int, default = 0,
                    help = 'random seed (default: 0)')

parser.add_argument('--src_path',type=str,default=p.join(PATH,'datasets'))
parser.add_argument('--dst_path', type=str, default=p.join(PATH, 'results'))

args = vars(parser.parse_args())

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    print(args)
    main(args)