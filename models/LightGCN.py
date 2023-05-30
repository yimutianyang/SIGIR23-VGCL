import tensorflow as tf
import os, pdb
import sys
sys.path.append('./models/')
from PairWise_model import Base_CF


class LightGCN(Base_CF):
    def __init__(self, args, dataset):
        super(LightGCN, self).__init__(args, dataset)
        self.gcn_layer = args.gcn_layer
        adj_indices, adj_values, adj_shape = dataset._convert_csr_to_sparse_tensor_inputs(dataset._lightgcn_adj_matrix())
        self.adj_matrix = tf.SparseTensor(adj_indices, adj_values, adj_shape)
        self._build_graph()


    def _create_lightgcn_emb(self):
        ego_emb = tf.concat([self.user_latent_emb, self.item_latent_emb], axis=0)
        all_emb = [ego_emb]
        for _ in range(self.gcn_layer):
            tmp_emb = tf.sparse.sparse_dense_matmul(self.adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)
        all_emb = tf.stack(all_emb, axis=1)
        mean_emb = tf.reduce_mean(all_emb, axis=1, keepdims=False)
        out_user_emb, out_item_emb = tf.split(mean_emb, [self.num_user, self.num_item], axis=0)
        return out_user_emb, out_item_emb


    def _build_graph(self):
        with tf.name_scope('forward'):
            self.user_emb, self.item_emb = self._create_lightgcn_emb()
        with tf.name_scope('optimization'):
            self.ranking_loss, self.regu_loss, self.auc = self.compute_bpr_loss(self.user_emb, self.item_emb)
            self.loss = self.ranking_loss + self.regu_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)