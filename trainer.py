# %% pytorch
from matplotlib.pyplot import axis
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import faiss
import h5py
import pickle

from ipdb import set_trace

# public library
from datetime import datetime
import os
from os.path import exists, join, dirname
import numpy as np
from tqdm import tqdm
import json
import shutil
import importlib
import random

os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# private library
from options import FixRandom
from utils import light_log, cal_recall, schedule_device


class ContrastiveLoss(nn.Module):
    def __init__(self, margin) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb, pos_pair=True):                                                  # (1, D)
        if pos_pair:
            loss = 0.5 * (torch.norm(emb_a - emb, dim=1).pow(2))
        else:
            dis_D = torch.norm(emb_a - emb, dim=1)
            loss = 0.5 * (torch.clamp(self.margin - dis_D, min=0).pow(2))

        return loss


class QuadrupletLoss(nn.Module):
    def __init__(self, margin, margin2) -> None:
        super().__init__()
        device = torch.device("cuda")
        self.cri = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)
        self.cri2 = nn.TripletMarginLoss(margin=margin2, p=2, reduction='sum').to(device)

    def forward(self, emb_a, emb_p, emb_n, emb_n2):                                                # (1, D)
        loss1 = self.cri(emb_a, emb_p, emb_n)
        loss2 = self.cri(emb_a, emb_p, emb_n2)
        loss = loss1 + loss2
        return loss


class Trainer:
    def __init__(self, options) -> None:

        self.opt = options

        # r variables
        self.step = 0
        self.epoch = 0
        self.current_lr = 0
        self.best_recalls = [0, 0, 0]

        # seed
        fix_random = FixRandom(self.opt.seed)
        self.seed_worker = fix_random.seed_worker()

        # set device
        self.opt.cGPU = schedule_device()
        if self.opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda")
        torch.cuda.set_device(self.opt.cGPU)
        self.device = torch.device("cuda")
        print('{}:{}{}'.format('device', self.device, torch.cuda.current_device()))

        # make model
        assert self.opt.phase[:4] == 'test', 'only support test mode.'
        self.model = self.make_model()

        # make dataset
        self.make_dataset()

    def make_dataset(self):
        assert self.opt.phase[:4] == 'test', 'only support test mode.'
        self.dataset = importlib.import_module('tmp.models.{}'.format(self.opt.dataset))

        self.whole_test_set = self.dataset.get_whole_test_set(self.opt)
        self.whole_test_data_loader = DataLoader(dataset=self.whole_test_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
    
    def make_model(self):
        '''build model
        '''
        assert self.opt.phase[:4] == 'test', 'only support test mode.'
        # build and load teacher or student net
        assert self.opt.resume != '', 'you need to define a resume path'
        if exists('tmp'):
            shutil.rmtree('tmp')
        os.mkdir('tmp')
        shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
        network = importlib.import_module('tmp.models.{}'.format(self.opt.net))
        model = network.deliver_model(self.opt, self.opt.phase[-3:]).to(self.device)
        checkpoint = torch.load(self.opt.resume)
        model.load_state_dict(checkpoint['state_dict'])

        print('{}:{}, {}:{}, {}:{}'.format(model.id, self.opt.net, 'loss', self.opt.loss, 'mu_dim', self.opt.output_dim, 'sigma_dim', self.opt.sigma_dim if self.opt.phase[-3:] == 'stu' else '-'))

        if self.opt.nGPU > 1:
            model = nn.DataParallel(model)

        return model

    def test(self):
        recalls, _ = self.get_recall(self.model, for_post_eval=True)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]))

        return recalls

  
    def get_recall(self, model, for_post_eval=False):
        model.eval()

        eval_dataloader = self.whole_test_data_loader
        eval_set = self.whole_test_set

        whole_mu = torch.zeros((len(eval_set), self.opt.output_dim), device=self.device)           # (N, D)
        whole_var = torch.zeros((len(eval_set), self.opt.sigma_dim), device=self.device)           # (N, D)
        gt = eval_set.get_positives()                                                              # (N, n_pos)

        with torch.no_grad():
            for iteration, (input, indices) in enumerate(tqdm(eval_dataloader), 1):
                input = input.to(self.device)
                mu, var = model(input)           # (B, D)

                # var = torch.exp(var)
                whole_mu[indices, :] = mu
                whole_var[indices, :] = var
                del input, mu, var

        n_values = [1, 5, 10]

        whole_var = torch.exp(whole_var)
        whole_mu = whole_mu.cpu().numpy()
        whole_var = whole_var.cpu().numpy()
        mu_q = whole_mu[eval_set.dbStruct.numDb:].astype('float32')
        mu_db = whole_mu[:eval_set.dbStruct.numDb].astype('float32')
        sigma_q = whole_var[eval_set.dbStruct.numDb:].astype('float32')
        sigma_db = whole_var[:eval_set.dbStruct.numDb].astype('float32')
        faiss_index = faiss.IndexFlatL2(mu_q.shape[1])
        faiss_index.add(mu_db)
        dists, preds = faiss_index.search(mu_q, max(n_values))                                     # the results is sorted

        # cull queries without any ground truth positives in the database
        val_inds = [True if len(gt[ind]) != 0 else False for ind in range(len(gt))]
        val_inds = np.array(val_inds)
        mu_q = mu_q[val_inds]
        sigma_q = sigma_q[val_inds]
        preds = preds[val_inds]
        dists = dists[val_inds]
        gt = gt[val_inds]

        recall_at_k = cal_recall(preds, gt, n_values)

        if for_post_eval:
            with open(join(self.opt.runsPath, 'embs.pickle'), 'wb') as handle:
                pickle.dump(mu_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(mu_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sigma_q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(sigma_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(whole_mu, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(whole_var, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('embeddings saved for post processing')

        return recall_at_k, None

