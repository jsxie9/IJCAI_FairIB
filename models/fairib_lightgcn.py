
import torch
import torch.nn as nn

from models.lightgcn import LightGCN


class FairIB_LightGCN(nn.Module):

    def __init__(self, n_users, n_items, norm_adj, emb_size, n_layers, device):
        super(FairIB_LightGCN, self).__init__()
        self.device = device
        self.norm_adj = norm_adj
        self.inter_enc = LightGCN(n_users, n_items, norm_adj, emb_size, n_layers, device)

    def forward(self):
        main_user_emb, main_item_emb, all_emb_list, mean_embeddings = self.inter_enc.forward()
        mean_item_emb = torch.sparse.mm(self.norm_adj.to(self.device), mean_embeddings)

        return main_user_emb, main_item_emb, all_emb_list, mean_item_emb