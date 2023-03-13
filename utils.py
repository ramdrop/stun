from collections import namedtuple
from os.path import join

import faiss
import numpy as np
from pynvml import *
from scipy import stats
from scipy.io import loadmat
from scipy.optimize import least_squares
from skimage import io


def linear_fit(x, y, w, report_error=False):
    def cost(p, x, y, w):
        k = p[0]
        b = p[1]
        error = y - (k * x + b)
        error *= w
        return error

    p_init = np.array([-1, 1])
    ret = least_squares(cost, p_init, args=(x, y, w), verbose=0)
    # print(ret['x'][0], ret['x'][1], )
    y_fitted = ret['x'][0] * x + ret['x'][1]
    error = ret['cost']
    if report_error:
        return y_fitted, error
    else:
        return y_fitted


def reduce_sigma(sigma, std_or_sq, log_or_linear, hmean_or_mean):
    ''' 
    input sigma: sigma^2, ([1, D])
    output sigma: sigma, (1)
    '''
    if log_or_linear == 'log':
        print('log')
        sigma = np.log(sigma)
    elif log_or_linear == 'linear':
        pass
    else:
        raise NameError('undefined')

    if std_or_sq == 'std':
        sigma = np.sqrt(sigma)
    elif std_or_sq == 'sq':
        pass
    else:
        raise NameError('undefined')

    if hmean_or_mean == 'hmean':
        sigma = stats.hmean(sigma, axis=1)       # ([numQ,])
    elif hmean_or_mean == 'mean':
        sigma = np.mean(sigma, axis=1)           # ([numQ,])
    else:
        raise NameError('undefined')

    return sigma


def schedule_device():
    ''' output id of the graphic card with most free memory
    '''
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    frees = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        # print("GPU", i, ":", nvmlDeviceGetName(handle))
        info = nvmlDeviceGetMemoryInfo(handle)
        frees.append(info.free / 1e9)
    nvmlShutdown()
    # print(frees)
    id = frees.index(max(frees))
    # print(id)
    return id

def light_log(path, args):
    with open(join(path, 'screen.log'), 'a') as f:
        for arg in args:
            f.write(arg)
            f.flush()
            print(arg, end='')


def cal_recall_from_embeddings(gt, qFeat, dbFeat):
    n_values = [1, 5, 10]

    # ---------------------------------------------------- sklearn --------------------------------------------------- #
    # knn = NearestNeighbors(n_jobs=-1)
    # knn.fit(dbFeat)
    # dists, predictions = knn.kneighbors(qFeat, len(dbFeat))

    # --------------------------------- use faiss to do NN search -------------------------------- #
    faiss_index = faiss.IndexFlatL2(qFeat.shape[1])
    faiss_index.add(dbFeat)
    dists, predictions = faiss_index.search(qFeat, max(n_values))                                  # the results is sorted

    recall_at_n = cal_recall(predictions, gt, n_values)
    return recall_at_n


def cal_recall(ranks, pidx, ks):

    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):
        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break

    recall_at_k /= ranks.shape[0]

    return recall_at_k * 100.0


def cal_apk(pidx, rank, k):
    if len(rank) > k:
        rank = rank[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(rank):
        if p in pidx and p not in rank[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(pidx), k) * 100.0


def cal_mapk(ranks, pidxs, k):

    return np.mean([cal_apk(a, p, k) for a, p in zip(pidxs, ranks)])


def get_zoomed_bins(sigma, num_of_bins):
    s_min = np.min(sigma)
    s_max = np.max(sigma)
    print(s_min, s_max)
    bins_parent = np.linspace(s_min, s_max, num=num_of_bins)
    k = 0
    while True:
        indices = []
        bins_child = np.linspace(bins_parent[0], bins_parent[-1 - k], num=num_of_bins)
        for index in range(num_of_bins - 1):
            target_q_ind_l = np.where(sigma >= bins_child[index])
            if index != num_of_bins - 2:
                target_q_ind_r = np.where(sigma < bins_child[index + 1])
            else:
                target_q_ind_r = np.where(sigma <= bins_child[index + 1])
            target_q_ind = np.intersect1d(target_q_ind_l[0], target_q_ind_r[0])
            indices.append(target_q_ind)
        # if len(indices[-1]) > int(sigma.shape[0] * 0.0005):
        if len(indices[-1]) > int(sigma.shape[0] * 0.001) or k == num_of_bins - 2:
            break
        else:
            k = k + 1
    # print('{:.3f}'.format(sum([len(x) for x in indices]) / sigma.shape[0]), [len(x) for x in indices])
    # print('k=', k)
    return indices, bins_child, k


def bin_pr(preds, dists, gt, vis=False):
    # dists_m = np.around(dists[:, 0], 2)          # (4620,)
    # dists_u = np.array(list(set(dists_m)))
    # dists_u = np.sort(dists_u)                   # small > large

    dists_u = np.linspace(np.min(dists[:, 0]), np.max(dists[:, 0]), num=100)

    recalls = []
    precisions = []
    for th in dists_u:
        TPCount = 0
        FPCount = 0
        FNCount = 0
        TNCount = 0
        for index_q in range(dists.shape[0]):
            # Positive
            if dists[index_q, 0] < th:
                # True
                if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    TPCount += 1
                else:
                    FPCount += 1
            else:
                if np.any(np.in1d(preds[index_q, 0], gt[index_q])):
                    FNCount += 1
                else:
                    TNCount += 1
        assert TPCount + FPCount + FNCount + TNCount == dists.shape[0], 'Count Error!'
        if TPCount + FNCount == 0 or TPCount + FPCount == 0:
            # print('zero')
            continue
        recall = TPCount / (TPCount + FNCount)
        precision = TPCount / (TPCount + FPCount)
        recalls.append(recall)
        precisions.append(precision)
    if vis:
        from matplotlib import pyplot as plt
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.plot(recalls, precisions)
        ax.set_title('Precision-Recall')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.savefig('pr.png', dpi=200)
    return recalls, precisions



def parse_dbStruct_pitts(path):
    dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset', 'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ', 'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    dataset = 'pitts'

    whichSet = matStruct[0].item()

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    dbImage = [f[0].item() for f in matStruct[1]]
    # dbImage = matStruct[1]
    utmDb = matStruct[2].T
    # utmDb = matStruct[2]

    # .mat file is generated by python, I replace the use of cell (in Matlab) with char (in Python)
    qImage = [f[0].item() for f in matStruct[3]]
    # qImage = matStruct[3]
    utmQ = matStruct[4].T
    # utmQ = matStruct[4]

    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr, posDistSqThr, nonTrivPosDistSqThr)

def cal_hs(img_path):
    img = io.imread(img_path, as_gray=True).reshape(-1, 1)
    counts, bins = np.histogram((img * 255).astype(np.int16), np.arange(0, 256, 1))
    counts = counts / np.sum(counts)
    cumulative = np.cumsum(counts)
    in_min = np.min((img*255).astype(np.int16))
    in_max = np.max((img*255).astype(np.int16))
    per_75 = np.argwhere(cumulative < 0.75)[-1]
    per_25 = np.argwhere(cumulative < 0.25)[-1]
    hs = (per_75 - per_25)/255
    return hs

if __name__ == '__main__':
    pass
