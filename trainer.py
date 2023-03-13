# %%
import importlib
import os
import pickle
import shutil
from os.path import dirname, exists, join
import h5py
import faiss
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import json
import torch.optim as optim

os.sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from options import FixRandom
from utils import cal_recall, light_log, schedule_device


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
        self.time_stamp = datetime.now().strftime('%m%d_%H%M%S')

        # set device
        if self.opt.phase == 'train_tea':
            self.opt.cGPU = schedule_device()
        if self.opt.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda :(")
        torch.cuda.set_device(self.opt.cGPU)
        self.device = torch.device("cuda")
        print('{}:{}{}'.format('device', self.device, torch.cuda.current_device()))

        # make model
        if self.opt.phase == 'train_tea':
            self.model, self.optimizer, self.scheduler, self.criterion = self.make_model()
        elif self.opt.phase == 'train_stu':
            self.teacher_net, self.student_net, self.optimizer, self.scheduler, self.criterion = self.make_model()
            self.model = self.teacher_net
        elif self.opt.phase in ['test_tea', 'test_stu']:
            self.model = self.make_model()
        else:
            raise Exception('Undefined phase :(')
        
        # make folders
        self.make_folders()
        # make dataset
        self.make_dataset()
        # online logs
        if self.opt.phase in ['train_tea', 'train_stu']:
            wandb.init(project="STUN", config=vars(self.opt), name=f"{self.opt.loss}_{self.opt.phase}_{self.time_stamp}")


    def make_folders(self):
        ''' create folders to store tensorboard files and a copy of networks files
        '''
        if self.opt.phase in ['train_tea', 'train_stu']:
            self.opt.runsPath = join(self.opt.logsPath, f"{self.opt.loss}_{self.opt.phase}_{self.time_stamp}")
            if not os.path.exists(join(self.opt.runsPath, 'models')):
                os.makedirs(join(self.opt.runsPath, 'models'))

            if not os.path.exists(join(self.opt.runsPath, 'transformed')):
                os.makedirs(join(self.opt.runsPath, 'transformed'))

            for file in [__file__, 'datasets/{}.py'.format(self.opt.dataset), 'networks/{}.py'.format(self.opt.net)]:
                shutil.copyfile(file, os.path.join(self.opt.runsPath, 'models', file.split('/')[-1]))

            with open(join(self.opt.runsPath, 'flags.json'), 'w') as f:
                f.write(json.dumps({k: v for k, v in vars(self.opt).items()}, indent=''))

    def make_dataset(self):
        ''' make dataset
        '''
        if self.opt.phase in ['train_tea', 'train_stu']:
            assert os.path.exists(f'datasets/{self.opt.dataset}.py'), 'Cannot find ' + f'{self.opt.dataset}.py :('
            self.dataset = importlib.import_module('datasets.' + self.opt.dataset)
        elif self.opt.phase in ['test_tea', 'test_stu']:
            self.dataset = importlib.import_module('tmp.models.{}'.format(self.opt.dataset))

        # for emb cache
        self.whole_train_set = self.dataset.get_whole_training_set(self.opt)
        self.whole_training_data_loader = DataLoader(dataset=self.whole_train_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        self.whole_val_set = self.dataset.get_whole_val_set(self.opt)
        self.whole_val_data_loader = DataLoader(dataset=self.whole_val_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        self.whole_test_set = self.dataset.get_whole_test_set(self.opt)
        self.whole_test_data_loader = DataLoader(dataset=self.whole_test_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=self.opt.cuda, worker_init_fn=self.seed_worker)
        # for train tuples
        if self.opt.loss == 'quad':
            self.train_set = self.dataset.get_quad_set(self.opt, self.opt.margin, self.opt.margin2)
            self.training_data_loader = DataLoader(dataset=self.train_set, num_workers=8, batch_size=self.opt.batchSize, shuffle=True, collate_fn=self.dataset.collate_quad_fn, worker_init_fn=self.seed_worker)
        else:
            self.train_set = self.dataset.get_training_query_set(self.opt, self.opt.margin)
            self.training_data_loader = DataLoader(dataset=self.train_set, num_workers=8, batch_size=self.opt.batchSize, shuffle=True, collate_fn=self.dataset.collate_fn, worker_init_fn=self.seed_worker)
        print('{}:{}, {}:{}, {}:{}, {}:{}, {}:{}'.format('dataset', self.opt.dataset, 'database', self.whole_train_set.dbStruct.numDb, 'train_set', self.whole_train_set.dbStruct.numQ, 'val_set', self.whole_val_set.dbStruct.numQ, 'test_set',
                                                    self.whole_test_set.dbStruct.numQ))
        print('{}:{}, {}:{}'.format('cache_bs', self.opt.cacheBatchSize, 'tuple_bs', self.opt.batchSize))


    def make_model(self):
        '''build model
        '''
        if self.opt.phase == 'train_tea':
            # build teacher net
            assert os.path.exists(f'networks/{self.opt.net}.py'), 'Cannot find ' + f'{self.opt.net}.py :('
            network = importlib.import_module('networks.' + self.opt.net)
            model = network.deliver_model(self.opt, 'tea')
            model = model.to(self.device)
            outputs = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))
            self.opt.output_dim = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[0].shape[-1]
            self.opt.sigma_dim = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[1].shape[-1]               # place holder
        elif self.opt.phase == 'train_stu':                                                                                                                          # load teacher net
            assert self.opt.resume != '', 'You need to define the teacher/resume path :('
            if exists('tmp'):
                shutil.rmtree('tmp')
            os.mkdir('tmp')
            shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
            network = importlib.import_module(f'tmp.models.{self.opt.net}')
            model_tea = network.deliver_model(self.opt, 'tea').to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model_tea.load_state_dict(checkpoint['state_dict'])
            # build student net
            assert os.path.exists(f'networks/{self.opt.net}.py'), 'Cannot find ' + f'{self.opt.net}.py :('
            network = importlib.import_module('networks.' + self.opt.net)
            model = network.deliver_model(self.opt, 'stu').to(self.device)
            self.opt.output_dim = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[0].shape[-1]
            self.opt.sigma_dim = model(torch.rand((2, 3, self.opt.height, self.opt.width), device=self.device))[1].shape[-1]
        elif self.opt.phase in ['test_tea', 'test_stu']:
            # load teacher or student net
            assert self.opt.resume != '', 'You need to define a teacher/resume path :('
            if exists('tmp'):
                shutil.rmtree('tmp')
            os.mkdir('tmp')
            shutil.copytree(join(dirname(self.opt.resume), 'models'), join('tmp', 'models'))
            network = importlib.import_module('tmp.models.{}'.format(self.opt.net))
            model = network.deliver_model(self.opt, self.opt.phase[-3:]).to(self.device)
            checkpoint = torch.load(self.opt.resume)
            model.load_state_dict(checkpoint['state_dict'])

        print('{}:{}, {}:{}, {}:{}'.format(model.id, self.opt.net, 'loss', self.opt.loss, 'mu_dim', self.opt.output_dim, 'sigma_dim', self.opt.sigma_dim if self.opt.phase[-3:] == 'stu' else '-'))

        if self.opt.phase in ['train_tea', 'train_stu']:
            # optimizer
            if self.opt.optim == 'adam':
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), self.opt.lr, weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, self.opt.lrGamma, last_epoch=-1, verbose=False)
            elif self.opt.optim == 'sgd':
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.weightDecay)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lrStep, gamma=self.opt.lrGamma)
            else:
                raise NameError('Undefined optimizer :(')

            # loss function
            if self.opt.loss == 'tri':
                criterion = nn.TripletMarginLoss(margin=self.opt.margin, p=2, reduction='sum').to(self.device)
            elif self.opt.loss == 'cont':
                criterion = ContrastiveLoss(margin=torch.tensor(self.opt.margin, device=self.device))
            elif self.opt.loss == 'quad':
                criterion = QuadrupletLoss(margin=self.opt.margin, margin2=self.opt.margin2).to(self.device)

        if self.opt.nGPU > 1:
            model = nn.DataParallel(model)

        if self.opt.phase == 'train_tea':
            return model, optimizer, scheduler, criterion
        elif self.opt.phase == 'train_stu':
            return model_tea, model, optimizer, scheduler, criterion
        elif self.opt.phase in ['test_tea', 'test_stu']:
            return model
        else:
            raise NameError('Undefined phase :(')


    def build_embedding_cache(self):
        '''build embedding cache, such that we can find the corresponding (p) and (n) with respect to (a) in embedding space
        '''
        self.train_set.cache = os.path.join(self.opt.runsPath, self.train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(self.train_set.cache, mode='w') as h5:
            h5feat = h5.create_dataset("features", [len(self.whole_train_set), self.opt.output_dim], dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader), 1):
                    input = input.to(self.device)                                                  # torch.Size([32, 3, 154, 154]) ([32, 5, 3, 200, 200])
                    emb, _ = self.model(input)
                    h5feat[indices.detach().numpy(), :] = emb.detach().cpu().numpy()
                    del input, emb

    def process_batch(self, batch_inputs):
        '''
        process a batch of input
        '''
        if self.opt.loss == 'quad':
            anchor, positives, negatives, negatives2, neg_counts, indices = batch_inputs
        else:
            anchor, positives, negatives, neg_counts, indices = batch_inputs

        # in case we get an empty batch
        if anchor is None:
            return None, None

        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor, where N = batchSize * (nQuery + nPos + n_neg)
        B = anchor.shape[0]                                                                        # ([8, 1, 3, 200, 200])
        n_neg = torch.sum(neg_counts)                                                              # tensor(80) = torch.sum(torch.Size([8]))
        if self.opt.loss == 'quad':
            input = torch.cat([anchor, positives, negatives, negatives2])                          # ([B, C, H, 200])
        else:
            input = torch.cat([anchor, positives, negatives])                                      # ([B, C, H, 200])

        input = input.to(self.device)            # ([96, 1, C, H, W])
        embs, vars = self.model(input)           # ([96, D])

        # monitor uncertainty values
        if self.step % 100 == 0:
            wandb.log({'sigma_sq/avg': torch.mean(vars).item()}, step=self.step)
            wandb.log({'sigma_sq/max': torch.max(vars).item()}, step=self.step)
            wandb.log({'sigma_sq/min': torch.min(vars).item()}, step=self.step)

        tuple_loss = 0
        # Standard triplet loss (via PyTorch library)
        if self.opt.loss == 'tri':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1])
            tuple_loss /= n_neg.float().to(self.device)
        # Contrastive loss
        elif self.opt.loss == 'cont':
            embs_a, embs_p, embs_n = torch.split(embs, [B, B, n_neg])                              # embs_a: ([B, D])
            dis_pos_min, dis_neg_min, dis_neg_avg = 0, 0, 0
            for i, neg_count in enumerate(neg_counts):
                dis_pos_min += torch.norm(embs_a[i:i + 1] - embs_p[i:i + 1], dim=1)
                tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], pos_pair=True)
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    if n == 0:
                        dis_neg_min += torch.norm(embs_a[i:i + 1] - embs_n[negIx:negIx + 1], dim=1)
                        dis_neg_avg += dis_neg_min
                    else:
                        dis_neg_avg += torch.norm(embs_a[i:i + 1] - embs_n[negIx:negIx + 1], dim=1)

                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_n[negIx:negIx + 1], pos_pair=False)
            tuple_loss /= (n_neg + 1).float().to(self.device)
            if self.step % 100 == 0:
                wandb.log({'pair_dis/pos_min': dis_pos_min.item()}, step=self.step)
                wandb.log({'pair_dis/neg_min': (dis_neg_min / n_neg).item()}, step=self.step)
                wandb.log({'pair_dis/neg_avg': (dis_neg_avg / (n_neg + 1)).item()}, step=self.step)
        # Quadruplet loss
        elif self.opt.loss == 'quad':
            embs_a, embs_p, embs_n, embs_n2 = torch.split(embs, [B, B, n_neg, n_neg])
            for i, neg_count in enumerate(neg_counts):
                for n in range(neg_count):
                    negIx = (torch.sum(neg_counts[:i]) + n).item()
                    tuple_loss += self.criterion(embs_a[i:i + 1], embs_p[i:i + 1], embs_n[negIx:negIx + 1], embs_n2[negIx:negIx + 1])
            tuple_loss /= 2 * n_neg.float().to(self.device)

        del input, embs, embs_a, embs_p, embs_n
        del anchor, positives, negatives

        return tuple_loss, n_neg

    def train(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # build embedding cache
            if self.epoch % self.opt.cacheRefreshEvery == 0:
                self.model.eval()
                self.build_embedding_cache()
                self.model.train()

            # train
            tuple_loss_sum = 0
            for _, batch_inputs in enumerate(tqdm(self.training_data_loader)):
                self.step += 1

                self.optimizer.zero_grad()
                tuple_loss, n_neg = self.process_batch(batch_inputs)
                if tuple_loss is None:
                    continue
                tuple_loss.backward()
                self.optimizer.step()
                tuple_loss_sum += tuple_loss.item()

                if self.step % 10 == 0:
                    wandb.log({'train_tuple_loss': tuple_loss.item()}, step=self.step)
                    wandb.log({'train_batch_num_neg': n_neg}, step=self.step)

            n_batches = len(self.training_data_loader)
            wandb.log({'train_avg_tuple_loss': tuple_loss_sum / n_batches}, step=self.step)
            torch.cuda.empty_cache()
            self.scheduler.step()

            # val every x epochs
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.val(self.model)
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    not_improved += self.opt.evalEvery
                # light log
                vars_to_log = [
                    'e={:>2d},'.format(self.epoch),
                    'lr={:>.8f},'.format(self.current_lr),
                    'tl={:>.4f},'.format(tuple_loss_sum / n_batches),
                    'r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]),
                    '\n' if not_improved else ' *\n',
                ]
                light_log(self.opt.runsPath, vars_to_log)
            else:
                recalls = None
            self.save_model(self.model, is_best=not not_improved)

            # stop when not improving for a period
            if self.opt.phase == 'train_tea':
                if self.opt.patience > 0 and not_improved > self.opt.patience:
                    print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                    break

        self.save_model(self.model, is_best=False)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1], self.best_recalls[2]))

        return self.best_recalls

    def train_student(self):
        not_improved = 0
        for epoch in range(self.opt.nEpochs):
            self.epoch = epoch
            self.current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            mu_delta_sq_sum, sigma_sq_sum, log_sigma_sq_sum, left_sum, loss_sum = 0, 0, 0, 0, 0
            n_batches = len(self.whole_training_data_loader)
            for iteration, (input, indices) in enumerate(tqdm(self.whole_training_data_loader)):
                self.step += 1
                input = input.to(self.device)    # ([B, C, H, W])
                self.optimizer.zero_grad()

                with torch.no_grad():
                    mu_tea, _ = self.teacher_net(input)                                            # ([B, D])
                mu_stu, log_sigma_sq = self.student_net(input)                                     # ([B, D]), ([B, D])

                # ---------------------- shift sigma_sq ---------------------- #
                if self.opt.loss in ['tri', 'quad']:    # empically found shifting distribution to be helpful for these losses
                    log_sigma_sq = torch.clamp(10 * log_sigma_sq + 0.2, 0, 1)
                # == numerator
                mu_delta = torch.norm((mu_stu - mu_tea), p=2, dim=-1, keepdim=True)                # L2 norm -> ([B, D])
                # == denominator
                sigma_sq = torch.exp(log_sigma_sq)
                # == regulizer
                loss = (mu_delta / sigma_sq + log_sigma_sq).mean()                                 # ([B, D])

                loss.backward()
                self.optimizer.step()

                mu_delta_sq_sum += mu_delta.mean().item()
                sigma_sq_sum += sigma_sq.mean().item()
                log_sigma_sq_sum += log_sigma_sq.mean().item()
                left_sum += (mu_delta / sigma_sq).mean().item()
                loss_sum += loss.item()
                if self.step % 5 == 0:
                    wandb.log({'student/loss_mu_delta_sq': mu_delta.mean().item()}, step=self.step)
                    wandb.log({'student/loss_sigma_sq': sigma_sq.mean().item()}, step=self.step)
                    wandb.log({'student/loss_log_sigma_sq': log_sigma_sq.mean().item()}, step=self.step)
                    wandb.log({'student/loss_left': (mu_delta / sigma_sq).mean().item()}, step=self.step)
                    wandb.log({'student/loss': loss.item()}, step=self.step)

            wandb.log({'student/epoch_loss_mu_delta_sq': mu_delta_sq_sum / n_batches}, step=self.step)
            wandb.log({'student/epoch_loss_sigma_sq': sigma_sq_sum / n_batches}, step=self.step)
            wandb.log({'student/epoch_loss_log_sigma_sq': log_sigma_sq_sum / n_batches}, step=self.step)
            wandb.log({'student/epoch_loss_left': left_sum / n_batches}, step=self.step)
            wandb.log({'student/epoch_loss': loss_sum / n_batches}, step=self.step)
            self.scheduler.step()

            # val
            if (self.epoch % self.opt.evalEvery) == 0:
                recalls = self.val(self.student_net)
                if recalls[0] > self.best_recalls[0]:
                    self.best_recalls = recalls
                    not_improved = 0
                else:
                    not_improved += self.opt.evalEvery

                light_log(self.opt.runsPath, [
                    f'e={self.epoch:>2d},',
                    f'lr={self.current_lr:>.8f},',
                    f'tl={loss_sum / n_batches:>.4f},',
                    f'r@1/5/10={recalls[0]:.2f}/{recalls[1]:.2f}/{recalls[2]:.2f}',
                    '\n' if not_improved else ' *\n',
                ])
            else:
                recalls = None

            self.save_model(self.student_net, is_best=False, save_every_epoch=True)
            if self.opt.patience > 0 and not_improved > self.opt.patience:
                print('terminated because performance has not improve for', self.opt.patience, 'epochs')
                break

        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(self.best_recalls[0], self.best_recalls[1], self.best_recalls[2]))
        return self.best_recalls

    def val(self, model):
        recalls, _ = self.get_recall(model)
        for i, n in enumerate([1, 5, 10]):
            wandb.log({'{}/{}_r@{}'.format(model.id, self.opt.split, n): recalls[i]}, step=self.step)
            # self.writer.add_scalar('{}/{}_r@{}'.format(model.id, self.opt.split, n), recalls[i], self.epoch)

        return recalls
    
    def test(self):
        recalls, _ = self.get_recall(self.model, save_embs=True)
        print('best r@1/5/10={:.2f}/{:.2f}/{:.2f}'.format(recalls[0], recalls[1], recalls[2]))

        return recalls

    def save_model(self, model, is_best=False, save_every_epoch=False):
        if is_best:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_best.pth.tar'))

        if save_every_epoch:
            torch.save({
                'epoch': self.epoch,
                'step': self.step,
                'state_dict': model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            }, os.path.join(self.opt.runsPath, 'ckpt_e_{}.pth.tar'.format(self.epoch)))

    def get_recall(self, model, save_embs=False):
        model.eval()

        if self.opt.split == 'val':
            eval_dataloader = self.whole_val_data_loader
            eval_set = self.whole_val_set
        elif self.opt.split == 'test':
            eval_dataloader = self.whole_test_data_loader
            eval_set = self.whole_test_set
        # print(f"{self.opt.split} len:{len(eval_set)}")

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

        if save_embs:
            with open(join(self.opt.runsPath, '{}_db_embeddings_{}.pickle'.format(self.opt.split, self.opt.resume.split('.')[-3].split('_')[-1])), 'wb') as handle:
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
