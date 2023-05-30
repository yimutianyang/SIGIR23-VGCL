import numpy as np
import tensorflow as tf
import sys, os, pdb
sys.path.append('./models/')
from PairWise_model import Base_CF


class SimGCL(Base_CF):
    def __init__(self, args, dataset):
        super(SimGCL, self).__init__(args, dataset)
        self.gcn_layer = args.gcn_layer
        self.eps = args.eps
        self.ssl_reg = args.ssl_reg
        self.ssl_temp = args.ssl_temp
        adj_indices, adj_values, adj_shape = dataset._convert_csr_to_sparse_tensor_inputs(dataset._lightgcn_adj_matrix())
        self.adj_matrix = tf.SparseTensor(adj_indices, adj_values, adj_shape)
        self._bulid_graph()

    def _lightgcn_encoder(self, emb):
        '''
        discard 0-th layer emb, refer to: SIGIR'22 SimGCL
        '''
        all_emb = []
        for i in range(self.gcn_layer):
            emb = tf.sparse.sparse_dense_matmul(self.adj_matrix, emb)
            all_emb.append(emb)
        all_emb = tf.reduce_mean(all_emb, axis=0)
        out_user_emb, out_item_emb = tf.split(all_emb, [self.num_user, self.num_item], axis=0)
        return out_user_emb, out_item_emb

    def _perturbed_lightgcn_encoder(self, emb):
        all_emb = []
        for i in range(self.gcn_layer):
            emb = tf.sparse.sparse_dense_matmul(self.adj_matrix, emb)
            random_noise = tf.random.uniform(emb.shape)
            emb += tf.multiply(tf.sign(emb), tf.nn.l2_normalize(random_noise, 1)) * self.eps
        all_emb.append(emb)
        all_emb = tf.reduce_mean(all_emb, axis=0)
        out_user_emb, out_item_emb = tf.split(all_emb, [self.num_user, self.num_item], axis=0)
        return out_user_emb, out_item_emb


    def _bulid_graph(self):
        with tf.name_scope('forward'):
            ego_emb = tf.concat([self.user_latent_emb, self.item_latent_emb], axis=0)
            self.user_emb, self.item_emb = self._lightgcn_encoder(ego_emb)
            self.user_emb_sub1, self.item_emb_sub1 = self._perturbed_lightgcn_encoder(ego_emb)
            self.user_emb_sub2, self.item_emb_sub2 = self._perturbed_lightgcn_encoder(ego_emb)
        with tf.name_scope('optimization'):
            self.ranking_loss, self.regu_loss, self.auc = self.compute_bpr_loss(self.user_emb, self.item_emb)
            self.ssl_loss = self.compute_ssl_loss()
            self.loss = self.ranking_loss + self.regu_loss + self.ssl_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def compute_variance(self):
        l2_dis = tf.square(self.user_emb_sub1 - self.user_emb_sub2)
        return tf.reduce_sum(l2_dis, 1, keepdims=True)

    def compute_ssl_loss(self):
        ###  user part  ###
        users = tf.unique(self.users)[0]
        items = tf.unique(self.pos_items)[0]
        user_emb1 = tf.nn.embedding_lookup(self.user_emb_sub1, users)
        user_emb2 = tf.nn.embedding_lookup(self.user_emb_sub2, users)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        normalize_all_user_emb2 = normalize_user_emb2
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        ssl_loss_user = -tf.reduce_mean(tf.log(pos_score_user / ttl_score_user))

        ###  item part  ###
        item_emb1 = tf.nn.embedding_lookup(self.item_emb_sub1, items)
        item_emb2 = tf.nn.embedding_lookup(self.item_emb_sub2, items)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        normalize_all_item_emb2 = normalize_item_emb2
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_a=False, transpose_b=True)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)
        ssl_loss_item = -tf.reduce_mean(tf.log((pos_score_item) / ttl_score_item))
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
        return ssl_loss