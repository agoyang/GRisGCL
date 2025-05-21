import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from torch_geometric.utils import get_laplacian
from torch import sparse_coo_tensor
import torch.nn.functional as F

class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                
                num_graph_node = rec_user_emb.shape[0] + rec_item_emb.shape[0]
                pos_item_idx_for_graph = torch.LongTensor(pos_idx) + rec_user_emb.shape[0]
                L_idx, L_weight = get_laplacian(torch.LongTensor([user_idx, pos_item_idx_for_graph]).cuda(), normalization=None, num_nodes=num_graph_node)
                L = sparse_coo_tensor(L_idx, L_weight, size=(num_graph_node,
                                                            num_graph_node))
                graph_emb = torch.cat((rec_user_emb, rec_item_emb), dim=0)
                
                pos_loss = torch.trace( graph_emb.T @ (L @ graph_emb))
                
                neg_item_idx_for_graph = torch.LongTensor(neg_idx) + rec_user_emb.shape[0]
                L_neg_idx, L_neg_weight = get_laplacian(torch.LongTensor([user_idx, neg_item_idx_for_graph]).cuda(), normalization=None, num_nodes=num_graph_node)
                L_neg = sparse_coo_tensor(L_neg_idx, L_neg_weight, size=(num_graph_node,
                                                            num_graph_node))
                neg_loss = torch.trace( graph_emb.T @ (L_neg @ graph_emb))
                coles_loss = (pos_loss - 0.9 * neg_loss) / num_graph_node
                t = 2

                cos_sim_u = F.cosine_similarity(user_emb, user_emb)
                cos_sim_in = F.cosine_similarity(neg_item_emb, neg_item_emb)
                cos_sim_ip = F.cosine_similarity(pos_item_emb, pos_item_emb)
                hom_loss = torch.mean(torch.exp(t * cos_sim_u)) + torch.mean(torch.exp(t * cos_sim_in)) + torch.mean(torch.exp(t * cos_sim_ip))
                het_loss = torch.mean(torch.exp(- t * torch.norm(pos_item_emb - neg_item_emb, dim=1) ** 2))
                regulation_loss = het_loss + hom_loss
                batch_loss = coles_loss + regulation_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb



    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        user_all_embeddings = F.normalize(user_all_embeddings)
        item_all_embeddings = F.normalize(item_all_embeddings)
        return user_all_embeddings, item_all_embeddings


