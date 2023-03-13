#%%
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from os.path import join
import utils
import importlib

importlib.reload(utils)

# --------------------------------------------------------------------------------------------------------------------- #
NETWORK = 'teacher_triplet'
# Choose NETWORK from |'teacher_triplet'|'student_contrast'|'student_triplet'|'student_quadruplet'|
# --------------------------------------------------------------------------------------------------------------------- #

NUM_BINS = 11
SHOW_AP = True
exp = NETWORK
resume = join('logs', NETWORK)

with open(join(resume, 'embs.pickle'), 'rb') as handle:
    q_mu = pickle.load(handle)
    db_mu = pickle.load(handle)
    q_sigma_sq = pickle.load(handle)
    db_sigma_sq = pickle.load(handle)
    preds = pickle.load(handle)
    dists = pickle.load(handle)
    gt = pickle.load(handle)
    _ = pickle.load(handle)
    _ = pickle.load(handle)


q_sigma_sq_h = np.mean(q_sigma_sq, axis=1)
db_sigma_sq_h = np.mean(db_sigma_sq, axis=1)
indices, _, _ = utils.get_zoomed_bins(q_sigma_sq_h, NUM_BINS)

# ---------------------- ECE and Recognition performance --------------------- #
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

    # calculate r@N
    recall_at_n = utils.cal_recall(pred_bin, gt_bin, n_values)
    bins_recall[index] = recall_at_n
    ece_bins_recall[index] = np.array([len(indices[index]) / q_sigma_sq_h.shape[0] * np.abs(recall_at_n[i] / 100.0 - (NUM_BINS - 1 - index) / ((NUM_BINS - 1))) for i in range(len(n_values))])

    # calculate mAP@N
    map_n = [utils.cal_mapk(pred_bin, gt_bin, n) for n in n_values]
    bins_map[index] = map_n
    ece_bins_map[index] = np.array([len(indices[index]) / q_sigma_sq_h.shape[0] * np.abs(map_n[i] / 100.0 - (NUM_BINS - 1 - index) / ((NUM_BINS - 1))) for i in range(len(n_values))])


# ---------------------------- uncertainty metric ---------------------------- #
print('ECE_rec@1/5/10: {:.3f} / {:.3f} / {:.3f}'.format(ece_bins_recall.sum(axis=0)[0], ece_bins_recall.sum(axis=0)[1], ece_bins_recall.sum(axis=0)[2]))
print('ECE_mAP@1/5/10: {:.3f} / {:.3f} / {:.3f}'.format(ece_bins_map.sum(axis=0)[0], ece_bins_map.sum(axis=0)[1], ece_bins_map.sum(axis=0)[2]))
if SHOW_AP:
    print('ECE_AP: {:.3f}'.format(ece_bins_ap.sum()))

# ---------------------------- recognition metric ---------------------------- #
recall = utils.cal_recall(preds, gt, n_values) / 100.0
print('rec@1/5/10: {:.3f} / {:.3f} / {:.3f}'.format(recall[0], recall[1], recall[2]))
map = [utils.cal_mapk(preds, gt, n) / 100.0 for n in n_values]
print('mAP@1/5/10: {:.3f} / {:.3f} / {:.3f}'.format(map[0], map[1], map[2]))

if SHOW_AP:
    recalls, precisions = utils.bin_pr(preds, dists, gt)
    ap = 0
    for index_j in range(len(recalls) - 1):
        ap += precisions[index_j] * (recalls[index_j + 1] - recalls[index_j])

    print('AP: {:.3f}'.format(ap))



# ------------------------------- visulization ------------------------------- #
w = np.array([len(indices[index]) / q_sigma_sq_h.shape[0] for index in range(NUM_BINS - 1)])
x = np.arange(0, NUM_BINS - 1, 1)

plt.style.use('ggplot')
fig, axs = plt.subplots(2, 2, figsize=(10, 10), squeeze=False)
fig.suptitle(exp)

ax = axs[0][0]
ax.bar(np.arange(len(indices)), [len(x) for x in indices])
ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
ax.set_ylabel('num of samples')

ax = axs[0][1]
ax.plot(np.arange(NUM_BINS - 1), bins_recall[:, 0], marker='o')
ax.plot(np.arange(NUM_BINS - 1), bins_recall[:, 1], marker='o')
ax.plot(np.arange(NUM_BINS - 1), bins_recall[:, 2], marker='o')

ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
ax.set_ylabel('recall@n')

ax = axs[1][0]
ax.plot(np.arange(NUM_BINS - 1), bins_map[:, 0], marker='o')
ax.plot(np.arange(NUM_BINS - 1), bins_map[:, 1], marker='o')
ax.plot(np.arange(NUM_BINS - 1), bins_map[:, 2], marker='o')

ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
ax.set_ylabel('mAP@n')

if SHOW_AP:
    ax = axs[1][1]
    ax.plot(np.arange(NUM_BINS - 1), bins_ap, marker='o')
    ax.set_xlabel('sigma^2\n(uncertainty: low -> high)')
    ax.set_ylabel('AP')
plt.savefig(join(resume, 'performance.png'), dpi=200)
