

import math
import scipy
import numpy as np
from collections import defaultdict


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    rec = hits / (1.0 * len(ground_list))
    return rec


def ndcg(ranked_list, ground_truth):
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg


def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg

def js_topk(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    return rank_js_distance, truth_rank_js_distance

def js_topk_multi(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    label_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1
        # label_rank_topk_items[uid][test_u2i[uid][:100]] = 1

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items
    rank_dis,truth_rank_dis, gt_rank_dis=[],[],[]
    for age in range(np.max(sens) + 1):
        indexx = sens == age
        rank_dis1 = np.sum(rank_topk_items[indexx], axis=0)
        truth_rank_dis1 = np.sum(truth_rank_topk_items[indexx], axis=0)
        rank_dis.append(rank_dis1)
        truth_rank_dis.append(truth_rank_dis1)

        # gt_rank_dis3 = np.sum(label_rank_topk_items[indexx], axis=0)
        # gt_rank_dis.append(gt_rank_dis3)

    dp_all,eo_all,gt_dp_all=[],[],[]
    for i in range(len(rank_dis)):
        for j in range(i + 1, len(rank_dis)):
            dp = scipy.spatial.distance.jensenshannon(rank_dis[i], rank_dis[j])
            dp_all.append(dp)
    for i in range(len(truth_rank_dis)):
        for j in range(i + 1, len(truth_rank_dis)):
            eo = scipy.spatial.distance.jensenshannon(truth_rank_dis[i], truth_rank_dis[j])
            eo_all.append(eo)

    return  np.mean(dp_all), np.mean(eo_all)

def js_topk_multi_gt(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    label_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1
        label_rank_topk_items[uid][test_u2i[uid][:100]] = 1

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items
    rank_dis,truth_rank_dis, gt_rank_dis=[],[],[]
    for age in range(np.max(sens) + 1):
        indexx = sens == age
        rank_dis1 = np.sum(rank_topk_items[indexx], axis=0)
        truth_rank_dis1 = np.sum(truth_rank_topk_items[indexx], axis=0)
        rank_dis.append(rank_dis1)
        truth_rank_dis.append(truth_rank_dis1)

        gt_rank_dis3 = np.sum(label_rank_topk_items[indexx], axis=0)
        gt_rank_dis.append(gt_rank_dis3)

    dp_all,eo_all,gt_dp_all=[],[],[]
    for i in range(len(rank_dis)):
        for j in range(i + 1, len(rank_dis)):
            dp = scipy.spatial.distance.jensenshannon(rank_dis[i], rank_dis[j])
            dp_all.append(dp)
    for i in range(len(truth_rank_dis)):
        for j in range(i + 1, len(truth_rank_dis)):
            eo = scipy.spatial.distance.jensenshannon(truth_rank_dis[i], truth_rank_dis[j])
            eo_all.append(eo)

    for i in range(len(gt_rank_dis)):
        for j in range(i + 1, len(gt_rank_dis)):
            gt_dp = scipy.spatial.distance.jensenshannon(gt_rank_dis[i], gt_rank_dis[j])
            gt_dp_all.append(gt_dp)



    return np.mean(gt_dp_all), np.mean(dp_all), np.mean(eo_all)
#增加了gt的dp
def js_topk2(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    label_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1
        label_rank_topk_items[uid][test_u2i[uid][:100]] = 1

    truth_rank_topk_items2 = truth_rank_topk_items & rank_topk_items

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.sum(truth_rank_topk_items2[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items2[index2], axis=0)

    truth_rank_dis3 = np.sum(label_rank_topk_items[index1], axis=0)
    truth_rank_dis4 = np.sum(label_rank_topk_items[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    label_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis3, truth_rank_dis4)

    return label_rank_js_distance, rank_js_distance, truth_rank_js_distance

#增加了 直接相减求和
def js_topk3(topk_items, sens, train_u2i, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    eod_rank_topk_items = np.ones((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist() #预测的
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1 # 每个用户预测的topk
        truth_rank_topk_items[uid][test_u2i[uid]] = 1 #gt=测试集中所有用户点击的
        eod_rank_topk_items[uid][test_u2i[uid]] = 0
        eod_rank_topk_items[uid][train_u2i[uid]] = 0 #无影响

    truth_rank_topk_items2 = truth_rank_topk_items & rank_topk_items # 喜欢的&预测的 #每个用户预测的topk且属于用户点击的
    eod_rank_topk_items2 = eod_rank_topk_items & rank_topk_items  # 不喜欢的&预测的topk

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.mean(rank_topk_items[index1], axis=0)
    rank_dis2 = np.mean(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.mean(truth_rank_topk_items2[index1], axis=0)
    truth_rank_dis2 = np.mean(truth_rank_topk_items2[index2], axis=0)

    # truth_rank_dis3 = np.sum(eod_rank_topk_items2[index1], axis=0)
    # truth_rank_dis4 = np.sum(eod_rank_topk_items2[index2], axis=0)
    truth_rank_dis31 = np.mean(eod_rank_topk_items2[index1], axis=0)
    truth_rank_dis41 = np.mean(eod_rank_topk_items2[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    # label_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis3, truth_rank_dis4)
    label_rank_js_distance2 = scipy.spatial.distance.jensenshannon(truth_rank_dis31, truth_rank_dis41)

    dp_abs = np.sum(np.abs(rank_dis1-rank_dis2))
    eo_abs = np.sum(np.abs(truth_rank_dis1 - truth_rank_dis2))
    return dp_abs, eo_abs, label_rank_js_distance2, rank_js_distance, truth_rank_js_distance

#增加pda的指标
def js_topk6(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    eod_rank_topk_items = np.ones((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1
        eod_rank_topk_items[uid][test_u2i[uid]] = 0

    truth_rank_topk_items2 = truth_rank_topk_items & rank_topk_items # 喜欢的&预测的
    eod_rank_topk_items2 = eod_rank_topk_items & rank_topk_items  # 不喜欢的&预测的

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.mean(rank_topk_items[index1], axis=0)
    rank_dis2 = np.mean(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.mean(truth_rank_topk_items2[index1], axis=0)
    truth_rank_dis2 = np.mean(truth_rank_topk_items2[index2], axis=0)


    truth_rank_dis31 = np.mean(eod_rank_topk_items2[index1], axis=0)
    truth_rank_dis41 = np.mean(eod_rank_topk_items2[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    label_rank_js_distance2 = scipy.spatial.distance.jensenshannon(truth_rank_dis31, truth_rank_dis41)

    dp_abs = np.sum(np.abs(rank_dis1-rank_dis2))
    eo_abs = np.sum(np.abs(truth_rank_dis1 - truth_rank_dis2))
    #fda metric
    rank_dis1_sum = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2_sum = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1_sum = np.sum(truth_rank_topk_items2[index1], axis=0)
    truth_rank_dis2_sum = np.sum(truth_rank_topk_items2[index2], axis=0)

    sum_dp = rank_dis1_sum+rank_dis2_sum
    dp=[]
    for index1, d in enumerate(sum_dp):
        if d >0:
            dp.append(abs(rank_dis1_sum[index1]-rank_dis2_sum[index1])/d)
    dp_fda = np.array(dp).mean()

    sum_eo = truth_rank_dis1_sum + truth_rank_dis2_sum
    eo = []
    for index1, d in enumerate(sum_eo):
        if d > 0:
            eo.append(abs(truth_rank_dis1_sum[index1] - truth_rank_dis2_sum[index1]) / d)
    eo_fda = np.array(eo).mean()
    return dp_fda, eo_fda, dp_abs, eo_abs, label_rank_js_distance2, rank_js_distance, truth_rank_js_distance


def js_topk8(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist() #预测的
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1 # 每个用户预测的topk
        truth_rank_topk_items[uid][test_u2i[uid]] = 1 #gt=测试集中所有用户点击的

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items #每个用户预测的topk且属于用户点击的

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0) #在topk中，并且用户点击了
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)
    #不在预测的topk=(1-rank_topk_items)，并且用户没点击(1-truth_rank_topk_items)
    truth_rank_topk_items2 = (1-truth_rank_topk_items) & (1-rank_topk_items)
    truth_rank_dis11 = np.sum(truth_rank_topk_items2[index1], axis=0)+truth_rank_dis1
    truth_rank_dis22 = np.sum(truth_rank_topk_items2[index2], axis=0)+truth_rank_dis2

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)
    truth_rank_js_distance2 = scipy.spatial.distance.jensenshannon(truth_rank_dis11, truth_rank_dis22)
    return truth_rank_js_distance2, rank_js_distance, truth_rank_js_distance

def js_topk10(topk_items, sens, train_u2i, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    eod_rank_topk_items = np.ones((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist() #预测的
    for uid in range(n_users):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1 # 每个用户预测的topk
        truth_rank_topk_items[uid][test_u2i[uid]] = 1 #gt=测试集中所有用户点击的
        eod_rank_topk_items[uid][test_u2i[uid]] = 0
        eod_rank_topk_items[uid][train_u2i[uid]] = 0 #无影响

    truth_rank_topk_items2 = truth_rank_topk_items & rank_topk_items # 喜欢的&预测的 #每个用户预测的topk且属于用户点击的
    eod_rank_topk_items2 = eod_rank_topk_items & rank_topk_items  # 不喜欢的&预测的topk

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.mean(rank_topk_items[index1], axis=0)
    rank_dis2 = np.mean(rank_topk_items[index2], axis=0)
    #oae
    oae_g1 = np.sum(truth_rank_topk_items2[index1]) / (topk*np.sum(index1))
    oae_g2 = np.sum(truth_rank_topk_items2[index2]) / (topk*np.sum(index2))
    oae = abs(oae_g1-oae_g2)

    #oae 2
    oae_g1 = np.sum(truth_rank_topk_items2[index1], axis=1)/topk#每个用户预测对的
    oae_g2 = np.sum(truth_rank_topk_items2[index2], axis=1) / topk  # 每个用户预测对的
    oae_g1 = np.mean(oae_g1)
    oae_g2 = np.mean(oae_g2)
    oae = abs(oae_g1 - oae_g2)

    #新EO，
    item_sum_g1 =np.sum(truth_rank_topk_items[index1], axis=0) #每个物品被多少用户喜欢
    item_sum_0_index_g1 = np.where(item_sum_g1 == 0)[0]
    item_sum_g2 = np.sum(truth_rank_topk_items[index2], axis=0)
    item_sum_0_index_g2 = np.where(item_sum_g2 == 0)[0]
    item_sum_0_index_all = np.array(list(set(item_sum_0_index_g1.tolist()).union(set(item_sum_0_index_g2.tolist()))))

    truth_rank_dis1 = np.sum(truth_rank_topk_items2[index1], axis=0)/(np.sum(truth_rank_topk_items[index1], axis=0)+1e-15)
    truth_rank_dis2 = np.sum(truth_rank_topk_items2[index2], axis=0)/(np.sum(truth_rank_topk_items[index2], axis=0)+1e-15)
    truth_rank_dis1 = np.delete(truth_rank_dis1, item_sum_0_index_all)
    truth_rank_dis2 = np.delete(truth_rank_dis2, item_sum_0_index_all)
    # end

    #eod
    # item_sum_g1 = np.sum(eod_rank_topk_items[index1], axis=0)
    # item_sum_0_index_g1 = np.where(item_sum_g1 == 0)[0]
    # item_sum_g2 = np.sum(eod_rank_topk_items[index2], axis=0)
    # item_sum_0_index_g2 = np.where(item_sum_g2 == 0)[0]
    # item_sum_0_index_all = np.array(list(set(item_sum_0_index_g1.tolist()).union(set(item_sum_0_index_g2.tolist()))))

    truth_rank_dis31 = np.sum(eod_rank_topk_items2[index1], axis=0) / (
                np.sum(eod_rank_topk_items[index1], axis=0) + 1e-15)
    truth_rank_dis41 = np.sum(eod_rank_topk_items2[index2], axis=0) / (
                np.sum(eod_rank_topk_items[index2], axis=0) + 1e-15)
    # truth_rank_dis31 = np.delete(truth_rank_dis1d, item_sum_0_index_all)
    # truth_rank_dis41 = np.delete(truth_rank_dis2d, item_sum_0_index_all)
    # truth_rank_dis3 = np.sum(eod_rank_topk_items2[index1], axis=0)
    # truth_rank_dis4 = np.sum(eod_rank_topk_items2[index2], axis=0)
    # truth_rank_dis31 = np.mean(eod_rank_topk_items2[index1], axis=0)
    # truth_rank_dis41 = np.mean(eod_rank_topk_items2[index2], axis=0)
    #end

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    # truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)
    truth_rank_js_distance = np.sum(np.abs(truth_rank_dis1-truth_rank_dis2))

    # label_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis3, truth_rank_dis4)
    label_rank_js_distance2 = scipy.spatial.distance.jensenshannon(truth_rank_dis31, truth_rank_dis41)

    dp_abs = np.sum(np.abs(rank_dis1-rank_dis2))
    eo_abs = np.sum(np.abs(truth_rank_dis1 - truth_rank_dis2))
    return oae, dp_abs, eo_abs, label_rank_js_distance2, rank_js_distance, truth_rank_js_distance
