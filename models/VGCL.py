import numpy as np
import tensorflow as tf
import sys, os, pdb
sys.path.append('./models/')
from PairWise_model import Base_CF


class VGCL(Base_CF):
    def __init__(self, args, dataset):
        super(VGCL, self).__init__(args, dataset)
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.gcn_layer = args.gcn_layer
        self.temp_node = args.temp_node
        self.temp_cluster = args.temp_cluster
        adj_indices, adj_values, adj_shape = dataset._convert_csr_to_sparse_tensor_inputs(dataset._lightgcn_adj_matrix())
        self.adj_matrix = tf.SparseTensor(adj_indices, adj_values, adj_shape)
        self.eps_weight = tf.Variable(self.nor_initializer([self.latent_dim, self.latent_dim]), name='eps_weight')
        self.eps_bias = tf.Variable(tf.zeros([self.latent_dim]), name='eps_bias')
        self.user_2cluster = tf.compat.v1.placeholder(tf.int32, [None, 1])
        self.item_2cluster = tf.compat.v1.placeholder(tf.int32, [None, 1])
        self._bulid_graph()


    def graph_encoder(self):
        '''
        Graph Inference Module
        Return: Gaussian distribution of each node N(mean, var)
        '''
        ego_emb = tf.concat([self.user_latent_emb, self.item_latent_emb], axis=0)
        all_emb = []
        for _ in range(self.gcn_layer):
            ego_emb = tf.sparse.sparse_dense_matmul(self.adj_matrix, ego_emb)
            all_emb.append(ego_emb)
        mean = tf.reduce_mean(all_emb, axis=0)
        logstd = tf.matmul(mean, self.eps_weight) + self.eps_bias
        std = tf.exp(logstd) * 0.01
        ### reparameterization
        noise1 = tf.random_normal(std.shape)
        noise2 = tf.random_normal(std.shape)
        noised_emb1 = mean + std * noise1
        noised_emb2 = mean + std * noise2
        return noised_emb1, noised_emb2, mean, std


    def kl_regulizer(self, mean, std):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        regu_loss = -0.5 * (1 + 2*std - tf.square(mean) - tf.square(tf.exp(std)))
        return tf.reduce_mean(tf.reduce_sum(regu_loss, 1, keepdims=True)) / self.args.batch_size


    def _bulid_graph(self):
        with tf.name_scope('forward'):
            noised_emb1, noised_emb2, self.mean, self.std = self.graph_encoder()
            self.user_emb, self.item_emb = tf.split(noised_emb1, [self.num_user, self.num_item], axis=0)
            self.user_emb_sub1, self.item_emb_sub1 = tf.split(noised_emb1, [self.num_user, self.num_item], axis=0)
            self.user_emb_sub2, self.item_emb_sub2 = tf.split(noised_emb2, [self.num_user, self.num_item], axis=0)
        with tf.name_scope('optimization'):
            self.ranking_loss, self.regu_loss, self.auc = self.compute_bpr_loss(self.user_emb, self.item_emb)
            self.regu_loss += self.reg_weight()
            self.cl_loss_node = self.compute_cl_loss_node()
            self.cl_loss_cluster = self.compute_cl_loss_cluster() * self.gamma
            self.cl_loss = self.cl_loss_node + self.cl_loss_cluster
            self.kl_loss = self.kl_regulizer(self.mean, self.std) * self.beta
            self.loss = self.ranking_loss + self.regu_loss + self.kl_loss + self.cl_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def reg_weight(self):
        reg_loss = tf.nn.l2_loss(self.eps_weight) + tf.nn.l2_loss(self.eps_bias)
        return self.l2_reg * reg_loss * 0.01

    def compute_cl_loss_node(self):
        '''
        Node-level contrastive learning
        '''
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
        pos_score_user = tf.exp(pos_score_user / self.temp_node)
        ttl_score_user = tf.reduce_sum(tf.exp(ttl_score_user / self.temp_node), axis=1)
        cl_loss_user = -tf.reduce_mean(tf.log(pos_score_user / ttl_score_user))

        ###  item part  ###
        item_emb1 = tf.nn.embedding_lookup(self.item_emb_sub1, items)
        item_emb2 = tf.nn.embedding_lookup(self.item_emb_sub2, items)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        normalize_all_item_emb2 = normalize_item_emb2
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_a=False, transpose_b=True)
        pos_score_item = tf.exp(pos_score_item / self.temp_node)
        ttl_score_item = tf.reduce_sum(tf.exp(ttl_score_item / self.temp_node), axis=1)
        cl_loss_item = -tf.reduce_mean(tf.log((pos_score_item) / ttl_score_item))
        cl_loss = self.alpha * (cl_loss_user + cl_loss_item)
        return cl_loss


    def compute_cl_loss_cluster(self):
        '''
        Cluster-level contrastive learning
        (1) K-means clustering as a special instance that prototype distribution is onehot
        (2) We select users/items with a same clustering prototype as the positive samples for each anchor node
        (3) Contrastive temperature can be assigned a smaller value compared to node-level cl loss
        '''
        ###  pos samples  ###
        users = tf.unique(self.users)[0]  # [m, ]
        items = tf.unique(self.pos_items)[0]
        user_cluster_id = tf.nn.embedding_lookup(self.user_2cluster, users)
        user_mask = tf.cast(tf.equal(user_cluster_id, tf.transpose(user_cluster_id)), tf.float32)  # [bs, bs]
        num_pos_per_cow = tf.reduce_sum(user_mask, axis=1)
        item_cluster_id = tf.nn.embedding_lookup(self.item_2cluster, items)
        item_mask = tf.cast(tf.equal(item_cluster_id, tf.transpose(item_cluster_id)), tf.float32)  # [bs, bs]
        num_item_pos_per_cow = tf.reduce_sum(item_mask, axis=1)

        ###  user contrastive learning  ###
        user_emb1 = tf.nn.embedding_lookup(self.user_emb_sub1, users)
        user_emb2 = tf.nn.embedding_lookup(self.user_emb_sub2, users)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        logit = tf.matmul(normalize_user_emb1, normalize_user_emb2, transpose_a=False, transpose_b=True)
        logit = logit / self.temp_cluster
        logit = logit - tf.reduce_max(tf.stop_gradient(logit), axis=1, keepdims=True)  ###去除每一行最大的元素
        exp_logit = tf.exp(logit)
        denominator = tf.reduce_sum(exp_logit, axis=1, keepdims=True)
        log_probs = exp_logit / denominator * user_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)
        log_probs = tf.math.divide_no_nan(log_probs, num_pos_per_cow)
        cl_loss_user = -tf.reduce_mean(tf.math.log(log_probs))

        ###  item contrastive learning  ###
        item_emb1 = tf.nn.embedding_lookup(self.item_emb_sub1, items)
        item_emb2 = tf.nn.embedding_lookup(self.item_emb_sub2, items)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        logit_item = tf.matmul(normalize_item_emb1, tf.transpose(normalize_item_emb2))
        logit_item = logit_item / self.temp_cluster
        logit_item = logit_item - tf.reduce_max(tf.stop_gradient(logit_item), axis=1, keepdims=True)
        exp_logit_item = tf.exp(logit_item)
        denominator_item = tf.reduce_sum(exp_logit_item, axis=1, keepdims=True)
        log_probs_item = exp_logit_item / denominator_item * item_mask
        log_probs_item = tf.reduce_sum(log_probs_item, axis=1)
        log_probs_item = tf.math.divide_no_nan(log_probs_item, num_item_pos_per_cow)
        cl_loss_item = -tf.reduce_mean(tf.math.log(log_probs_item))
        cl_loss = self.alpha * (cl_loss_user + cl_loss_item)
        return cl_loss
