#%%
from scipy import stats
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import join, dirname
import utils
import importlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default='/LOCAL/ramdrop/dataset/mmrec_dataset/7n5s_xy11')
parser.add_argument("--split", type=str, default='test', choices=['test', 'val'])
parser.add_argument("--network", type=str, default='res50')
parser.add_argument("--epoch", type=int)
args = parser.parse_args()

importlib.reload(utils)
# ------------------------------------- - ------------------------------------ #
DATASET = 'pitts'
NETWORK = args.network                           # 'res50'
LOSS = 'tri'                                     # |'cont'|'tri'|'quad'|

LOG_OR_LINEAR = 'linear'                         # |'linear'|'log'|
STD_OR_SQ = 'sq'                                 # |'std'|'sq'|
HMEAN_OR_MEAN = 'mean'                           # |'hmean'|'mean'|

NUM_BINS = 11
SHOW_AP = False
# ------------------------------------- - ------------------------------------ #

exp = '{}_{}_{}_{}_{}'.format(DATASET, NETWORK, LOG_OR_LINEAR, STD_OR_SQ, HMEAN_OR_MEAN)
resume = args.resume
print(resume)
with open(join(dirname(resume), '{}_db_embeddings_{}.pickle'.format(args.split, resume.split('.')[-3].split('_')[-1])), 'rb') as handle:
    q_mu = pickle.load(handle)
    db_mu = pickle.load(handle)
    q_sigma_sq = pickle.load(handle)
    db_sigma_sq = pickle.load(handle)
    preds = pickle.load(handle)
    dists = pickle.load(handle)
    gt = pickle.load(handle)
    _ = pickle.load(handle)
    _ = pickle.load(handle)

#%%
# CALCULATE ECE ====================== #
q_sigma_sq_h = utils.reduce_sigma(q_sigma_sq, STD_OR_SQ, LOG_OR_LINEAR, HMEAN_OR_MEAN)
indices, _, k = utils.get_zoomed_bins(q_sigma_sq_h, NUM_BINS)

bins_recall = np.zeros((NUM_BINS-1, 3))
bins_map = np.zeros((NUM_BINS-1, 3))
bins_ap = np.zeros((NUM_BINS - 1))

ece_bins_recall = np.zeros((NUM_BINS - 1, 3))
ece_bins_map = np.zeros((NUM_BINS - 1, 3))
ece_bins_ap = np.zeros((NUM_BINS - 1))

n_values = [1, 5, 10]
for index in tqdm(range(NUM_BINS - 1)):
    if len(indices[index]) == 0:
        continue

    pred_bin = preds[indices[index]]
    dist_bin = dists[indices[index]]
    gt_bin = gt[indices[index]]

    if SHOW_AP:
        # calculate AP
        recalls, precisions = utils.bin_pr(pred_bin, dist_bin, gt_bin)
        ap = 0
        for index_j in range(len(recalls) - 1):
            ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])
        bins_ap[index] = ap
        ece_bins_ap[index] = len(indices[index]) / q_sigma_sq_h.shape[0] * np.abs(ap - (NUM_BINS - 1 - index) / ((NUM_BINS - 1)))
        # ece_bins_ap[index] = np.abs(ap - (10 - index) * 0.1)

    # calculate r@N
    recall_at_n = utils.cal_recall(pred_bin, gt_bin, n_values)
    bins_recall[index] = recall_at_n
    ece_bins_recall[index] = np.array([len(indices[index]) / q_sigma_sq_h.shape[0] * np.abs(recall_at_n[i] / 100.0 - (NUM_BINS - 1 - index) / ((NUM_BINS - 1))) for i in range(len(n_values))])
    # ece_bins_recall[index] = np.array([np.abs(recall_at_n[i] / 100.0 - (10 - index) * 0.1) for i in range(len(n_values))])

    # calculate mAP@N
    map_n = [utils.cal_mapk(pred_bin, gt_bin, n) for n in n_values]
    bins_map[index] = map_n
    ece_bins_map[index] = np.array([len(indices[index]) / q_sigma_sq_h.shape[0] * np.abs(map_n[i] / 100.0 - (NUM_BINS - 1 - index) / ((NUM_BINS - 1))) for i in range(len(n_values))])
    # ece_bins_map[index] = np.array([np.abs(map_n[i] / 100.0 - (10 - index) * 0.1) for i in range(len(n_values))])


# PRINT SUMMARY ====================== #
# print('ECE_rec@1/5/10: {:.3f}/{:.3f}/{:.3f}'.format(ece_bins_recall.sum(axis=0)[0], ece_bins_recall.sum(axis=0)[1], ece_bins_recall.sum(axis=0)[2]))
# print('ECE_mAP@1/5/10: {:.3f}/{:.3f}/{:.3f}'.format(ece_bins_map.sum(axis=0)[0], ece_bins_map.sum(axis=0)[1], ece_bins_map.sum(axis=0)[2]))
# print('ECE_AP: {:.3f}'.format(ece_bins_ap.sum()))

#%%
# RECOGNITION METRIC ================= #
recall = utils.cal_recall(preds, gt, n_values) / 100.0
# print('rec@1/5/10: {:.3f}/{:.3f}/{:.3f}'.format(recall[0], recall[1], recall[2]))
map = [utils.cal_mapk(preds, gt, n) / 100.0 for n in n_values]
# print('mAP@1/5/10: {:.3f}/{:.3f}/{:.3f}'.format(map[0], map[1], map[2]))

if SHOW_AP:
    recalls, precisions = utils.bin_pr(preds, dists, gt)
    ap = 0
    for index_j in range(len(recalls) - 1):
        ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])
    # print('AP: {:.3f}'.format(ap))

#%%
# VISULIZATION ======================= #
w = np.array([len(indices[index]) / q_sigma_sq_h.shape[0] for index in range(NUM_BINS - 1)])
x = np.arange(0, NUM_BINS - 1, 1)

plt.style.use('ggplot')
fig, axs = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
fig.suptitle('k={}'.format(k))

ax = axs[0][0]
ax.bar(np.arange(len(indices)), [len(x) for x in indices])
ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
ax.set_ylabel('num of samples')

ax = axs[0][1]
ax.plot(np.arange(NUM_BINS - 1), bins_recall[:, 0], marker='o')
ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_recall[:, 0], w), marker='', alpha=0.2, c='black')
ax.plot(np.arange(NUM_BINS - 1), bins_recall[:, 1], marker='o')
ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_recall[:, 1], w), marker='', alpha=0.2, c='black')
ax.plot(np.arange(NUM_BINS - 1), bins_recall[:, 2], marker='o')
ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_recall[:, 2], w), marker='', alpha=0.2, c='black')

ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
ax.set_ylabel('recall@n')

ax = axs[1][0]
ax.plot(np.arange(NUM_BINS - 1), bins_map[:, 0], marker='o')
ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_map[:, 0], w), marker='', alpha=0.2, c='black')
ax.plot(np.arange(NUM_BINS - 1), bins_map[:, 1], marker='o')
ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_map[:, 1], w), marker='', alpha=0.2, c='black')
ax.plot(np.arange(NUM_BINS - 1), bins_map[:, 2], marker='o')
ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_map[:, 2], w), marker='', alpha=0.2, c='black')

ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
ax.set_ylabel('mAP@n')

if SHOW_AP:
    ax = axs[1][1]
    ax.plot(np.arange(NUM_BINS - 1), bins_ap, marker='o')
    ax.plot(np.arange(NUM_BINS - 1), utils.linear_fit(x, bins_ap, w), marker='', alpha=0.2, c='black')
    ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
    ax.set_ylabel('AP')


with open(join(dirname(resume),'{}.log'.format(args.split)), 'a') as f:
    arg = 'e:{:>2d} rec@1: {:.3f}'.format(args.epoch, recall[0])
    arg += ' ECE_rec@1/5/10: {:.3f}/{:.3f}/{:.3f}'.format(ece_bins_recall.sum(axis=0)[0], ece_bins_recall.sum(axis=0)[1], ece_bins_recall.sum(axis=0)[2])
    arg += ' ECE_mAP@1/5/10: {:.3f}/{:.3f}/{:.3f}\n'.format(ece_bins_map.sum(axis=0)[0], ece_bins_map.sum(axis=0)[1], ece_bins_map.sum(axis=0)[2])
    f.write(arg)
    f.flush()

plt.savefig(join(dirname(resume), 'ece_{}_{}.jpg'.format(args.split, args.epoch)), dpi=200)
