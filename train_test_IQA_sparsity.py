import os
import argparse
import random
import numpy as np
import sys
sys.path.append('.')
from HyperIQASolver import HyperIQASolver
import torch
import pickle

import sys
class Logger(object):
    def __init__(self, fileN="livec_bs16.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass


def fix_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main(config):

    folder_path = {config.dataset:config.dataset_path}

    expid = config.expid
    print('expid', expid)
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
    }
    sel_num = img_num[config.dataset]

    data_split_file = './CLIVE_traintest_index.pkl'
    if config.dataset == 'livec':
        with open(data_split_file, "rb") as f:
            sel_num = pickle.load(f)
    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)

    sys.stdout = Logger('livec_bs16.log')
    print('Dataset: %s, Batch Size: %i, If Grad: %s, Weight: %e.' % (config.dataset, config.batch_size, config.if_grad, config.weight))
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
        test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
        solver = HyperIQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='livec', help='Support datasets: livec') # For koniq-10k|bid|live|csiq|tid2013｜other datasets : generate corresponding data_split_file
    parser.add_argument('--dataset_path', dest='dataset_path', type=str, default='/YOURPATH/ChallengeDB_release/', help='path to livec')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')
    parser.add_argument('--expid', dest='expid', type=int, default=0, help='Exp ID')

    parser.add_argument('--if_grad', action="store_true", help='whether use l1 loss')
    parser.add_argument('--weight', type=float, default=1e-3, help='The weight of gradient loss')
    parser.add_argument('--h', type=float, default=1e-2, help='the step size when approximating the gradient')

    config = parser.parse_args()
    main(config)

