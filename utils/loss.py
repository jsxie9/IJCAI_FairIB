

import pdb
import torch
import torch.nn.functional as F



def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2

def calc_ib_loss(u_emb, sen_emb, sigma):
    Kx = kernel_matrix(u_emb, sigma)
    Ky = kernel_matrix(sen_emb, sigma)
    loss = hsic(Kx, Ky, u_emb.shape[0])
    return loss

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_score = torch.sum(user_emb * pos_emb, dim=1)
    neg_score = torch.sum(user_emb * neg_emb, dim=1)
    mf_loss = torch.mean(F.softplus(neg_score - pos_score))
    emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2)) / user_emb.shape[0]
    return mf_loss, emb_loss


def calc_bpr_loss(user_emb, item_emb, u, i, j):
    batch_user_emb = user_emb[u]
    batch_pos_item_emb = item_emb[i]
    batch_neg_item_emb = item_emb[j]

    mf_loss, emb_loss = bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
    return mf_loss, emb_loss



def mse_loss(user_emb, pos_emb, r):
    r_hat = (user_emb * pos_emb).sum(dim=-1)
    mf_loss = F.mse_loss(r_hat.squeeze(), r)
    emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2)) / user_emb.shape[0]
    return mf_loss, emb_loss


def calc_mse_loss(user_emb, item_emb, u, i, r):
    batch_user_emb = user_emb[u]
    batch_pos_item_emb = item_emb[i]
    mf_loss, emb_loss = mse_loss(batch_user_emb, batch_pos_item_emb, r)
    return mf_loss, emb_loss


