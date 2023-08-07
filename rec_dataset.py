import numpy as np
import random
import pdb
from collections import defaultdict
import scipy.sparse as sp
from time import time
import numba as nb
# import pymetis
import pdb
np.random.seed(2023)


@nb.njit()
def negative_sampling(training_user, training_item, traindata, num_item, num_negative):
    '''
    return: [u,i,j] for training, u interacted with i, not interacted with j
    '''
    trainingData = []
    for k in range(len(training_user)):
        u = training_user[k]
        pos_i = training_item[k]
        for _ in range(num_negative):
            neg_j = random.randint(0, num_item - 1)
            while neg_j in traindata[u]:
                neg_j = random.randint(0, num_item - 1)
            trainingData.append([u, pos_i, neg_j])
    return np.array(trainingData)


class CF_Dataset(object):
    def __init__(self, args):
        self.data_path = './datasets/'+args.dataset+'_data/'
        self.batch_size = args.batch_size
        self.traindata = np.load(self.data_path + 'traindata.npy', allow_pickle=True).tolist()
        self.testdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()
        self.num_user, self.num_item = self.max_user_and_item()
        #pdb.set_trace()
        self.num_node = self.num_user + self.num_item
        self.training_user, self.training_item = [], []
        for u, items in self.traindata.items():
            self.training_user.extend([u] * len(items))
            self.training_item.extend(items)
        ### nbdict就是把dict转成numba可用的形式，原理就是固定dict元素的类型为numpy;
        self.traindict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.traindata.items():
            if len(values) > 0:
                self.traindict[key] = np.asarray(list(values))
        self.testdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.testdata.items():
            if len(values) > 0:
                self.testdict[key] = np.asarray(list(values))


    def max_user_and_item(self):
        max_uid, max_iid = 0, 0
        for u, items in self.traindata.items():
            max_uid = max(max_uid, u)
            max_iid = max(max_iid, max(items))
        for u, items in self.testdata.items():
            max_uid = max(max_uid, u)
            max_iid = max(max_iid, max(items))
        return max_uid+1, max_iid+1


    def _user_group(self):
        u1, u2, u3, u4, u5 = [], [], [], [], []
        for u in self.testdata.keys():
            if u in self.traindata.keys():
                items = self.traindata[u]
            else:
                continue
            if len(items) < 8:
                u1.append(u)
            elif len(items) < 16:
                u2.append(u)
            elif len(items) < 32:
                u3.append(u)
            elif len(items) < 64:
                u4.append(u)
            else:
                u5.append(u)
        print('u1 size:', len(u1))
        print('u2 size:', len(u2))
        print('u3 size:', len(u3))
        print('u4 size:', len(u4))
        print('u5 size:', len(u5))
        return u1, u2, u3, u4, u5

    def _split_head_tail_data(self, ratio=0.8):
        user_degree, item_degree = [], []
        item_users = defaultdict(set)
        for u, items in self.traindata.items():
            user_degree.append([len(items), u])
            for i in items:
                item_users[i].add(u)
        for v, users in item_users.items():
            item_degree.append([len(users), v])
        user_degree = np.array(user_degree)
        item_degree = np.array(item_degree)
        topk_u = int(len(user_degree) * ratio)  ###长尾数目
        topk_v = int(len(item_degree) * ratio)  ###长尾数目
        user_sorted = np.argpartition(user_degree[:, 0], topk_u)
        head_user = user_degree[user_sorted[topk_u:]][:, 1]
        tail_user = set(range(self.num_user)) - set(head_user)
        item_sorted = np.argpartition(item_degree[:, 0], topk_v)
        head_item = item_degree[item_sorted[topk_v:]][:, 1]
        tail_item = set(range(self.num_item)) - set(head_item)
        return list(head_user), list(tail_user), list(head_item), list(tail_item)


    def _lightgcn_adj_matrix(self):
        '''
        return: sparse adjacent matrix, refer lightgcn
        '''
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    def _attack_lightgcn_adj_matrix(self):
        '''
        random add noise edges to original graph
        return: attacked sparse adjacent matrix
        this part not used in our VGCL paper
        '''
        user_np = np.array(self.training_user)
        item_np = np.array(self.training_item)
        users = np.arange(0, self.num_user)
        items = np.arange(0, self.num_item)
        attack_num = int(user_np.shape[0] * 0.1)
        attack_user = np.random.choice(users, attack_num, replace=True)
        attack_item = np.random.choice(items, attack_num, replace=True)
        user_np = np.concatenate([user_np, attack_user], axis=0)
        item_np = np.concatenate([item_np, attack_item], axis=0)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    def _corrupted_lightgcn_adj_matrix(self, drop_ratio=0):
        '''
        randomly drop edges, return adj matrix
        refer to: SIGIR21' SGL
        '''
        indexs = np.arange(len(self.training_user))
        keep_idx = np.random.choice(indexs, size=int(len(self.training_user) * (1 - drop_ratio)),
                                    replace=False)  # False表示无放回采样
        user_np = np.array(self.training_user)[keep_idx]
        item_np = np.array(self.training_item)[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_user)), shape=(self.num_node, self.num_node))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape


    def _batch_sampling(self, num_negative):
        t1 = time()
        ### 三元组采样使用numba加速
        triplet_data = negative_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                         self.traindict, self.num_item, num_negative)
        print('prepare training data cost time:{:.4f}'.format(time() - t1))
        batch_num = int(len(triplet_data) / self.batch_size) + 1
        indexs = np.arange(triplet_data.shape[0])
        np.random.shuffle(indexs)
        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indexs))
            if index_end == len(indexs):
                index_start = len(indexs) - self.batch_size
            batch_data = triplet_data[indexs[index_start:index_end]]
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
