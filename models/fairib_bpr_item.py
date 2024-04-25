
import torch
import torch.nn as nn



class FairIB_BPR_Item(nn.Module):

    def __init__(self, n_users, n_items, norm_adj, emb_size, n_layers, device):
        super(FairIB_BPR_Item, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj.to(device)
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.device = device
        self._init_embeddings()

    def _init_embeddings(self, ):
        self.embeddings = nn.ModuleDict()
        self.embeddings['user_embeddings'] = nn.Embedding(self.n_users, self.emb_size).to(self.device)
        self.embeddings['item_embeddings'] = nn.Embedding(self.n_items, self.emb_size).to(self.device)
        nn.init.xavier_uniform_(self.embeddings['user_embeddings'].weight)
        nn.init.xavier_uniform_(self.embeddings['item_embeddings'].weight)

    def forward(self):
        user_emb = self.embeddings['user_embeddings'].weight
        item_emb = self.embeddings['item_embeddings'].weight
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        # main_user_emb, main_item_emb, all_emb_list, mean_embeddings = self.inter_enc.forward()

        
        mean_item_emb = torch.sparse.mm(self.norm_adj.to(self.device), ego_embeddings)

        return user_emb, item_emb, mean_item_emb