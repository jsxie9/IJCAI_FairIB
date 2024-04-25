
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import datetime
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from models.fairib_lightgcn import FairIB_LightGCN
# from models.FairIB_LightGCN import FairIB_LightGCN
from utils import *
from models import *
from tqdm import tqdm

import scipy.stats as stats

import pdb
import sys

def train_semigcn(gcn, sens, n_users, lr=0.001, num_epochs=1000, device='cpu'):
    sens = torch.tensor(sens).to(torch.long).to(device)
    optimizer = optim.Adam(gcn.parameters(), lr=lr)

    final_loss = 0.0
    for _ in tqdm(range(num_epochs)):
        _, _, su, _ = gcn()
        shuffle_idx = torch.randperm(n_users)
        classify_loss = F.cross_entropy(su[shuffle_idx].squeeze(), sens[shuffle_idx].squeeze())
        optimizer.zero_grad()
        classify_loss.backward()
        optimizer.step()
        final_loss = classify_loss.item()

    print('epoch: %d, classify_loss: %.6f' % (num_epochs, final_loss))

def train_unify_mi(sens_enc, fair_ib, dataset, u_sens,
                   n_users, n_items, train_u2i, test_u2i, args):
    optimizer_G = optim.Adam(fair_ib.parameters(), lr=args.lr)
    u_sens_tensor = torch.tensor(u_sens, dtype=torch.long, device=args.device)

    e_su, e_si, _, _ = sens_enc.forward()
    e_su = e_su.detach().to(args.device)
    e_si = e_si.detach().to(args.device)
    e_su1 = e_su[u_sens_tensor.bool()]
    e_su0 = e_su[~u_sens_tensor.bool()]
    #calculate the mean value of sensitive representations of each user group
    e_su1_mean = torch.mean(e_su1, 0).unsqueeze(0)
    e_su0_mean = torch.mean(e_su0, 0).unsqueeze(0)
    e_su01_mean = torch.cat([e_su0_mean, e_su1_mean], dim=0)

    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    
    best_perf = 0.0
    early_stop = 0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'ib': 0.0,
        }

        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)#Positive item
            j = uij[2].type(torch.long).to(args.device)#negtive item
            main_user_emb, main_item_emb, all_emb_list, mean_emb = fair_ib.forward()
            
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg

            u_unique = torch.unique(u)
            e_zu, e_zi,_,_ = fair_ib.forward()
            u_emb = main_user_emb[u_unique]

            sen_code = u_sens_tensor[u_unique]
            sen_emb = e_su01_mean[sen_code]
            mean_user_emb, _ = torch.split(mean_emb, [n_users, n_items], dim=0)
            mean_user_emb = mean_user_emb[u_unique]
            #user-side + sub-graph side
            ib_loss = (args.beta * (calc_ib_loss(u_emb, sen_emb, args.sigma)) +
                       args.gamma * (calc_ib_loss(mean_user_emb, sen_emb, args.sigma)))
            loss = bpr_loss + emb_loss +  ib_loss
           
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr'] += bpr_loss.item()
            train_res['emb'] += emb_loss.item()
            train_res['ib'] += ib_loss.item()

        train_res['bpr'] = train_res['bpr'] / len(train_loader)
        train_res['emb'] = train_res['emb'] / len(train_loader)
        train_res['ib'] = train_res['ib'] / len(train_loader)

        a = datetime.datetime.now()
        time_str = datetime.datetime.strftime(a, "%m-%d %H:%M:%S ")

        training_logs = time_str+'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)
        early_stop += 1
        with torch.no_grad():
            t_user_emb, t_item_emb, _, _ = fair_ib.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens,
                num_workers=args.num_workers)

            p_eval = ''
            for keys, values in test_res.items():
                p_eval += keys + ':' + '[%.6f]' % values + ' '
            print(p_eval)

            if best_perf < test_res['ndcg@10']:
                early_stop = 0
                best_perf = test_res['ndcg@10']
                torch.save(fair_ib, args.param_path)
                print('save successful')
            # if early_stop > 50:
            #     print("early_stop")
            #     return

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_gcn_fairib',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='gcn')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_path', type=str, default='logs/ib_gcn_movie_')
    parser.add_argument('--param_path', type=str, default='param/fairib_lightgcn_')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--beta', type=float, default=40)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--sigma', type=float, default=0.3, help='gaosi kernel parameter')
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument('--lreg', type=float, default=0.1)

    args = parser.parse_args()

    set_seed(seed=args.seed)

    a = datetime.datetime.now()
    time_str = datetime.datetime.strftime(a, "%m-%d %H%M")
    pre_dex = "beta=" + str(args.beta) + "_gamma=" + str(args.gamma)+ "_sigma=" + str(args.sigma)
    args.log_path = args.log_path + pre_dex + " " + time_str + ".txt"
    sys.stdout = Logger(args.log_path)
    args.param_path = args.param_path + pre_dex + " " + time_str + ".pth"
    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()
    #Sensitive Attribute Encoder
    sens_enc = SemiGCN(n_users, n_items, norm_adj,
                       args.emb_size, args.n_layers, args.device,
                       nb_classes=np.unique(u_sens).shape[0])
    train_semigcn(sens_enc, u_sens, n_users, device=args.device)

    fair_ib = FairIB_LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    fair_ib.to(args.device)
    train_unify_mi(sens_enc,fair_ib, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args)
    sys.stdout = None
