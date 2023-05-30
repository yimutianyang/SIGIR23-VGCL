import numpy as np
import tensorflow as tf
import sys, os, pdb
sys.path.append('./models/')
from PairWise_model import Base_CF


class NCL(Base_CF):
    def __init__(self, args, dataset):
        super(NCL, self).__init__(args, dataset)
        self.gcn_layer = args.gcn_layer
        self.hyper_layer = args.hyper_layer
        self.ssl_reg = args.ssl_reg
        self.ssl_temp = args.ssl_temp
        self.alpha = args.alpha
        self.proto_reg = args.proto_reg
        adj_indices, adj_values, adj_shape = dataset._convert_csr_to_sparse_tensor_inputs(dataset._lightgcn_adj_matrix())
        self.adj_matrix = tf.SparseTensor(adj_indices, adj_values, adj_shape)
        self._create_variable()
        self._bulid_graph()

    def _create_variable(self):
        with tf.name_scope('input_cluster'):
            self.user_2cluster = tf.compat.v1.placeholder(tf.int32, [None, 1])
            self.item_2cluster = tf.compat.v1.placeholder(tf.int32, [None, 1])
            self.user_centroids = tf.compat.v1.placeholder(tf.float32, [None, self.latent_dim])
            self.user_centroids = tf.nn.l2_normalize(self.user_centroids, 1)
            self.item_centroids = tf.compat.v1.placeholder(tf.float32, [None, self.latent_dim])
            self.item_centroids = tf.nn.l2_normalize(self.item_centroids, 1)

    def _lightgcn_encoder(self, emb):
        all_emb = [emb]
        for i in range(self.gcn_layer):
            emb = tf.sparse.sparse_dense_matmul(self.adj_matrix, emb)
            all_emb.append(emb)
        mean_emb = tf.reduce_mean(all_emb, axis=0)
        out_user_emb, out_item_emb = tf.split(mean_emb, [self.num_user, self.num_item], axis=0)
        return out_user_emb, out_item_emb, all_emb



    def _bulid_graph(self):
        with tf.name_scope('forward'):
            ego_emb = tf.concat([self.user_latent_emb, self.item_latent_emb], axis=0)
            self.user_emb, self.item_emb, self.emb_list = self._lightgcn_encoder(ego_emb)
        with tf.name_scope('optimization'):
            self.ranking_loss, self.regu_loss, self.auc = self.compute_bpr_loss(self.user_emb, self.item_emb)
            self.ssl_loss = self.ssl_layer_loss()
            self.proto_loss = self.ssl_proto_loss()
            self.loss_warm = self.ranking_loss + self.regu_loss + self.ssl_loss
            self.loss = self.loss_warm + self.proto_loss
            self.opt_warm = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_warm)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)



    def ssl_layer_loss(self):
        '''
        structure contrastive learning
        '''
        user_emb_view1, item_emb_view1 = tf.split(self.emb_list[2*self.hyper_layer], [self.num_user, self.num_item], axis=0)
        user_emb_view2, item_emb_view2 = tf.split(self.emb_list[0], [self.num_user, self.num_item], axis=0)
        user_emb1 = tf.nn.embedding_lookup(user_emb_view1, self.users)
        user_emb2 = tf.nn.embedding_lookup(user_emb_view2, self.users)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        normalize_all_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))

        ###  item part  ###
        item_emb1 = tf.nn.embedding_lookup(item_emb_view1, self.pos_items)
        item_emb2 = tf.nn.embedding_lookup(item_emb_view2, self.pos_items)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        normalize_all_item_emb2 = tf.nn.l2_normalize(item_emb_view2, 1)
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_a=False, transpose_b=True)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)
        ssl_loss_item = -tf.reduce_sum(tf.log((pos_score_item) / ttl_score_item))
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item * self.alpha)
        return ssl_loss



    def ssl_proto_loss(self):
        '''
        semantic contrastive learning
        '''
        ###  user contrastive learning  ###
        user_emb1 = tf.nn.embedding_lookup(self.user_emb, self.users)
        user_2cluster = tf.nn.embedding_lookup(self.user_2cluster, self.users)
        user_emb2 = tf.gather_nd(self.user_centroids, user_2cluster)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = user_emb2
        normalize_all_user_emb2 = self.user_centroids
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_a=False, transpose_b=True)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.ssl_temp), axis=1)
        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))

        ###  item contrastive learning  ###
        item_emb1 = tf.nn.embedding_lookup(self.item_emb, self.pos_items)
        item_2cluster = tf.nn.embedding_lookup(self.item_2cluster, self.pos_items)
        item_emb2 = tf.gather_nd(self.item_centroids, item_2cluster)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = item_emb2
        normalize_all_item_emb2 = self.item_centroids
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_a=False, transpose_b=True)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.ssl_temp), axis=1)
        ssl_loss_item = -tf.reduce_sum(tf.log((pos_score_item) / ttl_score_item))
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
        return ssl_loss