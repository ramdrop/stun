import argparse
import json
import os
import random

import numpy as np
import torch
from genericpath import exists


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options")
        self.parser.add_argument('--phase', type=str, default='test_tea', help='phase', choices=['train_tea', 'test_tea', 'train_stu', 'test_stu'])
        self.parser.add_argument('--dataset', type=str, default='pitts', help='choose dataset.')
        self.parser.add_argument('--structDir', type=str, default='pittsburgh/structure', help='Path for structure.')
        self.parser.add_argument('--imgDir', type=str, default='pittsburgh', help='Path for images.')
        self.parser.add_argument('--com', type=str, default='', help='comment')
        self.parser.add_argument('--height', type=int, default=200, help='number of sequence to use.')
        self.parser.add_argument('--width', type=int, default=200, help='number of sequence to use.')
        self.parser.add_argument('--net', type=str, default='res50gem', help='network')
        self.parser.add_argument('--trainer', type=str, default='trainer', help='trainer')
        self.parser.add_argument('--loss', type=str, default='tri', help='triplet loss or bayesian triplet loss', choices=['tri', 'cont', 'quad'])
        self.parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
        self.parser.add_argument('--margin2', type=float, default=0.1, help='Margin2 for quadruplet loss. Default=0.1')
        self.parser.add_argument('--output_dim', type=int, default=0, help='Number of feature dimension. Default=512')
        self.parser.add_argument('--sigma_dim', type=int, default=0, help='Number of sigma dimension. Default=512')
        self.parser.add_argument('--batchSize', type=int, default=8, help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
        self.parser.add_argument('--cacheBatchSize', type=int, default=128, help='Batch size for caching and testing')
        self.parser.add_argument('--cacheRefreshRate', type=int, default=0, help='How often to refresh cache, in number of queries. 0 for off')
        self.parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
        self.parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
        self.parser.add_argument('--cGPU', type=int, default=2, help='core of GPU to use.')
        self.parser.add_argument('--optim', type=str, default='adam', help='optimizer to use', choices=['sgd', 'adam'])
        self.parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate.')
        self.parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
        self.parser.add_argument('--lrGamma', type=float, default=0.99, help='Multiply LR by Gamma for decaying.')
        self.parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
        self.parser.add_argument('--cuda', action='store_false', help='use cuda')
        self.parser.add_argument('--d', action='store_true', help='debug mode')
        self.parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
        self.parser.add_argument('--seed', type=int, default=1234, help='Random seed to use.')
        self.parser.add_argument('--logsPath', type=str, default='./logs', help='Path to save runs to.')
        self.parser.add_argument('--runsPath', type=str, default='not defined', help='Path to save runs to.')
        self.parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
        self.parser.add_argument('--evalEvery', type=int, default=1, help='Do a validation set run, and save, every N epochs.')
        self.parser.add_argument('--cacheRefreshEvery', type=int, default=1, help='refresh embedding cache, every N epochs.')
        self.parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
        self.parser.add_argument('--split', type=str, default='val', help='Split to use', choices=['val', 'test'])
        self.parser.add_argument('--encoder_dim', type=int, default=512, help='Number of feature dimension. Default=512')

    def parse(self):
        options = self.parser.parse_args()
        return options

    def update_opt_from_json(self, flag_file, options):
        if not exists(flag_file):
            raise ValueError('{} not exist'.format(flag_file))
        # restore_var = ['runsPath', 'net', 'seqLen', 'num_clusters', 'output_dim', 'structDir', 'imgDir', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 'num_clusters', 'optim', 'margin', 'seed', 'patience']
        do_not_update_list = ['resume', 'mode', 'phase', 'optim', 'split']
        if os.path.exists(flag_file):
            with open(flag_file, 'r') as f:
                # stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}
                stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k not in do_not_update_list}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in self.parser._actions:
                        if act.dest == flag[2:]:                                                   # stored parser match current parser
                                                                                                   # store_true / store_false args don't accept arguments, filter these
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                            else:
                                if val == str(act.default):
                                    to_del.append(flag)

                for flag, val in stored_flags.items():
                    missing = True
                    for act in self.parser._actions:
                        if flag[2:] == act.dest:
                            missing = False
                    if missing:
                        to_del.append(flag)

                for flag in to_del:
                    del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('restored flags:', train_flags)
                options = self.parser.parse_args(train_flags, namespace=options)
        return options


class FixRandom:
    def __init__(self, seed) -> None:
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def seed_worker(self):
        worker_seed = self.seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)
