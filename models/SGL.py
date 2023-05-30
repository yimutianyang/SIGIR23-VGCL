import tensorflow as tf
import os, sys, pdb
sys.path.append('./models/')
from PairWise_model import Base_CF

'''
Reproduction of SIGIR'21 SGL: Self-supervised Graph Learning for Recommendation
Source code: https://github.com/wujcan/SGL-TensorFlow
'''
class SGL(Base_CF):
    def __init__(self, args, dataset):
        super(SGL, self).__init__(args, dataset)
        self.gcn_layer = args.gcn_layer
        self.ssl_reg = args.ssl_reg
        self.ssl_temp = args.ssl_reg
        adj_indices, adj_values, adj_shape = dataset._convert_csr_to_sparse_tensor_inputs(dataset._lightgcn_adj_matrix())
        self.adj_matrix = tf.SparseTensor(adj_indices, adj_values, adj_shape)
        self._bulid_graph()


    def _create_variable(self):
        with tf.name_scope('input_adj_subgraph'):
            self.adj_indices_sub1 = tf.compat.v1.placeholder(tf.int64)
            self.adj_values_sub1 = tf.compat.v1.placeholder(tf.float32)
            self.adj_shape_sub1 = tf.compat.v1.placeholder(tf.int64)
            self.adj_indices_sub2 = tf.compat.v1.placeholder(tf.int64)
            self.adj_values_sub2 = tf.compat.v1.placeholder(tf.float32)
            self.adj_shape_sub2 = tf.compat.v1.placeholder(tf.int64)
            self.adj_matrix_sub1 = tf.SparseTensor(self.adj_indices_sub1, self.adj_values_sub1, self.adj_shape_sub1)
            self.adj_matrix_sub2 = tf.SparseTensor(self.adj_indices_sub2, self.adj_values_sub2, self.adj_shape_sub2)


    def _create_lightgcn_emb(self, _ego_emb, _adj_matrix):
        all_emb = [_ego_emb]
        for _ in range(self.gcn_layer):
            tmp_emb = tf.sparse.sparse_dense_matmul(_adj_matrix, all_emb[-1])
            all_emb.append(tmp_emb)
        all_emb = tf.stack(all_emb, axis=1)
        all_emb = tf.reduce_mean(all_emb, axis=1, keepdims=False)
        out_user_emb, out_item_emb = tf.split(all_emb, [self.num_user, self.num_item], axis=0)
        return out_user_emb, out_item_emb


    def _create_sgl_emb(self):
        ego_emb = tf.concat([self.user_latent_emb, self.item_latent_emb], axis=0)
        ego_emb_sub1 = ego_emb
        ego_emb_sub2 = ego_emb
        out_user_emb, out_item_emb = self._create_lightgcn_emb(ego_emb, self.adj_matrix)
        out_user_emb_sub1, out_item_emb_sub1 = self._create_lightgcn_emb(ego_emb_sub1, self.adj_matrix_sub1)
        out_user_emb_sub2, out_item_emb_sub2 = self._create_lightgcn_emb(ego_emb_sub2, self.adj_matrix_sub2)
        return out_user_emb, out_item_emb, out_user_emb_sub1, out_item_emb_sub1, out_user_emb_sub2, out_item_emb_sub2


    def _bulid_graph(self):
        self._create_variable()
        with tf.name_scope('forward'):
            self.user_emb, self.item_emb, self.user_emb_sub1, self.item_emb_sub1, \
            self.user_emb_sub2, self.item_emb_sub2 = self._create_sgl_emb()
        with tf.name_scope('optimization'):
            self.ranking_loss, self.regu_loss, self.auc = self.compute_bpr_loss(self.user_emb, self.item_emb)
            self.ssl_loss = self._ssl_loss()
            self.loss = self.ranking_loss + self.regu_loss + self.ssl_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def compute_variance(self):
        l2_dis = tf.square(self.user_emb_sub1 - self.user_emb_sub2)
        return tf.reduce_sum(l2_dis, 1, keepdims=True)


    def _ssl_loss(self):
        users = tf.unique(self.users)[0]
        items = tf.unique(self.pos_items)[0]
        ### user part ssl
        user_emb1 = tf.nn.embedding_lookup(self.user_emb_sub1, users)
        user_emb2 = tf.nn.embedding_lookup(self.user_emb_sub2, users)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        normalize_all_user_emb2 = tf.nn.l2_normalize(self.user_emb_sub2, 1)
        # normalize_all_user_emb2 = normalize_user_emb2
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        ssl_loss_user = -tf.reduce_mean(tf.log(pos_score_user / ttl_score_user))

        ### item part ssl
        item_emb1 = tf.nn.embedding_lookup(self.item_emb_sub1, items)
        item_emb2 = tf.nn.embedding_lookup(self.item_emb_sub2, items)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        normalize_all_item_emb2 = tf.nn.l2_normalize(self.item_emb_sub2, 1)
        # normalize_all_item_emb2 = normalize_item_emb2
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_a=False, transpose_b=True)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)
        ssl_loss_item = -tf.reduce_mean(tf.log(pos_score_item / ttl_score_item))
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
        return ssl_loss