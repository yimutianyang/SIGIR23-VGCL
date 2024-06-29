import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss


class VGCL(nn.Module):
    def __init__(self, args, dataset):
        super(VGCL, self).__init__()
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.gcn_layer = args.gcn_layer
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.init_type = args.init_type
        self.l2_reg = args.l2_reg
        self.alpha = args.alpha   ### cl loss cofficient
        self.beta = args.beta     ### KL term cofficient
        self.gamma = args.gamma   ### cluster loss cofficient
        self.temp_node = args.temp_node
        self.temp_cluster = args.temp_cluster
        self.num_cluster_user = args.num_user_cluster
        self.num_cluster_item = args.num_item_cluster
        self.eps_weight = torch.nn.Parameter(torch.randn(self.latent_dim, self.latent_dim), requires_grad=True)
        self.eps_bias = torch.nn.Parameter(torch.zeros(self.latent_dim), requires_grad=True)
        self.adj_matrix = dataset.get_ui_matrix()   # user-item interaction adjacent matrix
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self._init_weights()


    def _init_weights(self):
        self.user_embeddings = nn.Embedding(self.num_user, self.latent_dim)
        self.item_embeddings = nn.Embedding(self.num_item, self.latent_dim)
        if self.init_type == 'norm':
            nn.init.normal_(self.user_embeddings.weight, std=0.01)
            nn.init.normal_(self.item_embeddings.weight, std=0.01)
        elif self.init_type == 'xa_norm':
            nn.init.xavier_normal_(self.user_embeddings.weight)
            nn.init.xavier_normal_(self.item_embeddings.weight)
        else:
            raise NotImplementedError
        return None


    def graph_encoder(self):
        '''
        Graph Inference
        Return: Gaussian distribution of each node N(mean, std)
        '''
        ego_emb = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_emb = []
        for i in range(self.gcn_layer):
            ego_emb = torch.sparse.mm(self.adj_matrix, ego_emb)
            all_emb.append(ego_emb)
        mean = torch.mean(torch.stack(all_emb, dim=0), dim=0)
        logstd = torch.matmul(mean, self.eps_weight) + self.eps_bias
        std = torch.exp(logstd)
        return mean, std


    def reparameter(self, mean, std, scale=0.01):
        random_noise = torch.randn(std.shape).to(self.device)
        embedding = mean + std * random_noise * scale
        return embedding


    def forward(self):
        '''
        Node distribution inference
        Embedding reparameterization
        '''
        mean, std = self.graph_encoder()
        embedding_view1 = self.reparameter(mean, std)
        embedding_view2 = self.reparameter(mean, std)
        return mean, std, embedding_view1, embedding_view2


    def getEmbedding(self, users, pos_items, neg_items):
        rec_emb_users, rec_emb_items = torch.split(self.embeddings_view1, [self.num_user, self.num_item], dim=0)
        users_emb = rec_emb_users[users]
        pos_emb = rec_emb_items[pos_items]
        neg_emb = rec_emb_items[neg_items]
        users_emb_ego = self.user_embeddings(users)
        pos_emb_ego = self.item_embeddings(pos_items)
        neg_emb_ego = self.item_embeddings(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego


    def bpr_loss(self, users, pos_items, neg_items):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos_items.long(), neg_items.long())
        reg_loss = (userEmb0.norm(2).pow(2) +
                    posEmb0.norm(2).pow(2) +
                    negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        auc = torch.mean((pos_scores > neg_scores).float())
        bpr_loss = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
        return auc, bpr_loss, reg_loss*self.l2_reg


    def kl_regularizer(self, mean, std):
        '''
        KL term in ELBO loss
        Constraint approximate posterior distribution closer to prior
        '''
        regu_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        return regu_loss.sum(1).mean() / self.batch_size


    def compute_cl_loss_node(self, embeddings_view1, embeddings_view2, users, pos_items):
        users = torch.unique(users)
        items = torch.unique(pos_items)
        user_view1, item_view1 = torch.split(embeddings_view1, [self.num_user, self.num_item], dim=0)
        user_view2, item_view2 = torch.split(embeddings_view2, [self.num_user, self.num_item], dim=0)

        ### user part ###
        user_sub1, user_sub2 = user_view1[users], user_view2[users]
        user_sub1 = F.normalize(user_sub1, p=2, dim=1)
        user_sub2 = F.normalize(user_sub2, p=2, dim=1)
        pos_score_user = torch.multiply(user_sub1, user_sub2).sum(1) # [bs, 1]
        all_score_user = torch.matmul(user_sub1, user_sub2.transpose(0, 1)) # [bs, bs]
        pos_score_user = torch.exp(pos_score_user / self.temp_node) # [bs, 1]
        all_score_user = torch.exp(all_score_user / self.temp_node).sum(1) # [bs, 1]
        cl_loss_user = -torch.log(pos_score_user / all_score_user).mean()

        ### item part ###
        item_sub1, item_sub2 = item_view1[items], item_view2[items]
        item_sub1 = F.normalize(item_sub1, p=2, dim=1)
        item_sub2 = F.normalize(item_sub2, p=2, dim=1)
        pos_score_item = torch.multiply(item_sub1, item_sub2).sum(1)
        all_score_item = torch.matmul(item_sub1, item_sub2.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.temp_node)
        all_score_item = torch.exp(all_score_item / self.temp_node).sum(1)
        cl_loss_item = -torch.log(pos_score_item / all_score_item).mean()
        cl_loss_node = self.alpha * (cl_loss_user + cl_loss_item)
        return cl_loss_node


    def compute_cl_loss_cluster(self, embeddings_view1, embeddings_view2, users, pos_items, user_2cluster, item_2cluster):
        '''
        Cluster-level contrastive learning
        (1) K-means clustering as a special instance that prototype distribution is onehot
        (2) We select users/items with a same clustering prototype as the positive samples for each anchor node
        (3) Contrastive temperature can be assigned a smaller value compared to node-level cl loss
        '''
        ###  pos samples  ###
        users = torch.unique(users)  # [m, ]
        items = torch.unique(pos_items)
        user_cluster_id = user_2cluster[users]
        user_mask = torch.eq(user_cluster_id, user_cluster_id.transpose(0, 1)).float()
        avg_positive_user = user_mask.sum(dim=1)
        item_cluster_id = item_2cluster[items]
        item_mask = torch.eq(item_cluster_id, item_cluster_id.transpose(0,1)).float()  # [bs, bs]
        avg_positive_item = item_mask.sum(dim=1)
        user_view1, item_view1 = torch.split(embeddings_view1, [self.num_user, self.num_item], dim=0)
        user_view2, item_view2 = torch.split(embeddings_view2, [self.num_user, self.num_item], dim=0)

        ###  user contrastive learning  ###
        user_sub1, user_sub2 = user_view1[users], user_view2[users]
        user_sub1 = F.normalize(user_sub1, p=2, dim=1)
        user_sub2 = F.normalize(user_sub2, p=2, dim=1)
        logit = torch.matmul(user_sub1, user_sub2.transpose(0, 1))
        logit = logit - logit.detach().max(dim=1, keepdim=True)[0]  # 去除每一行最大的元素
        exp_logit = torch.exp(logit / self.temp_cluster)
        pos_score_user = (exp_logit * user_mask).sum(1) / avg_positive_user
        all_score_user = exp_logit.sum(1)
        cl_loss_user = -torch.log(pos_score_user / all_score_user).mean()

        ###  item contrastive learning  ###
        item_sub1, item_sub2 = item_view1[items], item_view2[items]
        item_sub1 = F.normalize(item_sub1, p=2, dim=1)
        item_sub2 = F.normalize(item_sub2, p=2, dim=1)
        logit = torch.matmul(item_sub1, item_sub2.transpose(0, 1))
        logit = logit - logit.detach().max(dim=1, keepdim=True)[0]  # 去除每一行最大的元素
        exp_logit = torch.exp(logit / self.temp_cluster)
        pos_score_item = (exp_logit * item_mask).sum(1) / avg_positive_item
        all_score_item = exp_logit.sum(1)
        cl_loss_item = -torch.log(pos_score_item / all_score_item).mean()
        cl_loss = self.alpha * (cl_loss_user + cl_loss_item)
        return cl_loss


    def run_kmeans(self, x, latent_dim, num_cluster):
        kmeans = faiss.Kmeans(d=latent_dim, k=num_cluster)
        kmeans.train(x)
        # cluster_cents = kmeans.centroids
        _, Index = kmeans.index.search(x, 1)
        #  convert to cuda Tensor for broadcasrt
        node2cluster = torch.LongTensor(Index).to(self.device)
        return node2cluster


    def cluster_step(self):
        user_embeddings = self.user_embeddings.weight.detach().cpu().numpy()
        item_embeddings = self.item_embeddings.weight.detach().cpu().numpy()
        user_2cluster = self.run_kmeans(user_embeddings, self.latent_dim, self.num_cluster_user)
        item_2cluster = self.run_kmeans(item_embeddings, self.latent_dim, self.num_cluster_item)
        return user_2cluster, item_2cluster


    def calculate_all_loss(self, users, pos_items, neg_items, user_2cluster, item_2cluster):
        # 1. learning embeddings based on graph encoder
        self.mean, self.std, self.embeddings_view1, self.embeddings_view2 = self.forward()
        # 2. calculate graph reconstruction loss, just bpr ranking loss and kl loss
        auc, bpr_loss, reg_loss = self.bpr_loss(users, pos_items, neg_items)
        kl_loss = self.kl_regularizer(self.mean, self.std)
        # 3. calculate node-level contrastive loss
        cl_loss_node = self.compute_cl_loss_node(self.embeddings_view1, self.embeddings_view2, users, pos_items)
        # 4. calculate cluster-level contrastive loss
        cl_loss_cluster = self.gamma * self.compute_cl_loss_cluster(self.embeddings_view1, self.embeddings_view2,
                                                       users, pos_items, user_2cluster, item_2cluster)
        cl_loss = cl_loss_node + cl_loss_cluster
        loss = bpr_loss + reg_loss + cl_loss + kl_loss
        return auc, bpr_loss, kl_loss, cl_loss_node, cl_loss_cluster, loss