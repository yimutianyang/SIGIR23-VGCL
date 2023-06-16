import tensorflow as tf
import numpy as np
import os, pdb, sys
from time import time
# sys.path.append('../')
from evaluate import *
from models.VGCL import VGCL
from rec_dataset import CF_Dataset
from tqdm import tqdm
from shutil import copyfile
from log import Logger
import faiss
import argparse
np.random.seed(2023)
tf.set_random_seed(2023)


def parse_args():
    ###  dataset parameters   ###
    parser = argparse.ArgumentParser(description='VGCL Parameters')
    parser.add_argument('--dataset', type=str, default='douban_book', help='which data to use')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negative samples')
    ###  training parameters  ###
    parser.add_argument('--device_id', type=int, default=0, help='CUDA ID')
    parser.add_argument('--log', type=str, default='True', help='write log or not?')
    parser.add_argument('--runid', type=int, default=0, help='current log id')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--topk', type=int, default=20, help='Topk value for evaluation')   # NDCG@20 as convergency metric
    parser.add_argument('--early_stops', type=int, default=5, help='model convergent when NDCG@20 not increase for x epochs')
    ###  model parameters  ###
    parser.add_argument('--gcn_layer', type=int, default=2, help='number of hidden layers in gcn encoder')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='l2 regularization for parameters')
    parser.add_argument('--alpha', type=float, default=0.2, help='contrastive learning loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='KL term weight for ELBO based loss')
    parser.add_argument('--gamma', type=float, default=0.4, help='cluster-level contrastive loss weight')
    parser.add_argument('--temp_node', type=float, default=0.2, help='temperature for node-level contrastive learning')
    parser.add_argument('--temp_cluster', type=float, default=0.13, help='temperature for cluster-aware contrastive learning')
    parser.add_argument('--num_user_cluster', type=int, default=900, help='number of user clusterings')
    parser.add_argument('--num_item_cluster', type=int, default=300, help='number of item clusterings')
    return parser.parse_args()


def run_kmeans(x, num_cluster):
    """Run K-means algorithm to get k clusters of the input tensor x
    """
    kmeans = faiss.Kmeans(d=x.shape[1], k=num_cluster, gpu=True)
    kmeans.train(x)
    cluster_cents = kmeans.centroids
    _, I = kmeans.index.search(x, 1)
    return cluster_cents, I


if __name__ == '__main__':
    ################################  record settings  ###################################
    args = parse_args()
    dataset_name = args.dataset
    runid = args.runid
    record_path = './saved/' + dataset_name + '_results/vgcl/'+'runid_'+str(runid)+'/'
    model_save_path = record_path + 'models/'
    print('model saved path is', model_save_path)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    copyfile('run_VGCL.py', record_path + 'run_VGCL.py')
    copyfile('./models/VGCL.py', record_path + 'VGCL.py')
    copyfile('./rec_dataset.py', record_path + 'rec_dataset.py')
    if args.log:
        log = Logger(record_path)
        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    ############################### build model and dataset  ##############################
    rec_data = CF_Dataset(args)
    rec_model = VGCL(args, rec_data)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)


    #********************************  start training  ************************************#
    writer = tf.summary.FileWriter(record_path+'/log/', sess.graph)
    max_hr, max_recall, max_ndcg, early_stop = 0, 0, 0, 0
    topk = args.topk
    for epoch in range(args.epochs):
        t1 = time()
        user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
        user_cents, I_u = run_kmeans(user_matrix, args.num_user_cluster)
        item_cents, I_i = run_kmeans(item_matrix, args.num_item_cluster)
        feed_dict = {rec_model.user_2cluster: I_u, rec_model.item_2cluster: I_i}
        data_iter = rec_data._batch_sampling(num_negative=args.num_neg)
        sum_auc, sum_loss1, sum_loss2, sum_loss3, sum_loss4, batch_num = 0, 0, 0, 0, 0, 0
        for batch_u, batch_i, batch_j in tqdm(data_iter):
            feed_dict.update({rec_model.users: batch_u, rec_model.pos_items: batch_i, rec_model.neg_items: batch_j})
            _auc, _loss1, _loss2, _loss3, _loss4, _ = sess.run([rec_model.auc, rec_model.ranking_loss, rec_model.kl_loss,
                                                    rec_model.cl_loss_node, rec_model.cl_loss_cluster, rec_model.opt], feed_dict=feed_dict)
            sum_auc += _auc
            sum_loss1 += _loss1
            sum_loss2 += _loss2
            sum_loss3 += _loss3
            sum_loss4 += _loss4
            batch_num += 1
        mean_auc = sum_auc / batch_num
        mean_loss1 = sum_loss1 / batch_num
        mean_loss2 = sum_loss2 / batch_num
        mean_loss3 = sum_loss3 / batch_num
        mean_loss4 = sum_loss4 / batch_num
        mean_loss = mean_loss1 + mean_loss2 + mean_loss3 + mean_loss4
        log.write('Epoch:{:d}, Train_auc:{:.4f}, Loss_rank:{:.4f}, Loss_kl:{:.4f}, Loss_ssl:{:.4f}, Loss_cluster:{:.4f}\n'
              .format(epoch, mean_auc, mean_loss1, mean_loss2, mean_loss3, mean_loss4))
        t2 = time()
        summary_train_loss = sess.run(rec_model.merged_train_loss, feed_dict={rec_model.train_loss: mean_loss,
                                                                              rec_model.train_mf_loss: mean_loss1})
        writer.add_summary(summary_train_loss, epoch)

        # ***************************  Evaluation on Top-20  *****************************#
        if epoch >= 0:
            early_stop += 1
            user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
            hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                                  [20], user_matrix, item_matrix, rec_data.testdata.keys())   ### all item evaluation
            log.write('Epoch:{:d}, topk:{:d}, recall:{:.4f}, ndcg{:.4f}\n'.format(epoch, topk, recall[20], ndcg[20]))
            rs = sess.run(rec_model.merged_evaluate,
                          feed_dict={rec_model.train_loss: mean_loss, rec_model.train_mf_loss: mean_loss1,
                                     rec_model.recall: recall[20], rec_model.ndcg: ndcg[20]})
            writer.add_summary(rs, epoch)
            max_hr = max(max_hr, hr[20])
            max_recall = max(max_recall, recall[20])
            max_ndcg = max(max_ndcg, ndcg[20])
            if ndcg[20] == max_ndcg:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[20]) + '.ckpt'
                saver.save(sess, model_save_path + best_ckpt)
            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            if epoch > 20 and early_stop > args.early_stops:
                log.write('early stop\n')
                log.write('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n'.format(max_recall, max_ndcg))
                break

    #***********************************  start evaluate testdata   ********************************#
    writer.close()
    saver.restore(sess, model_save_path + best_ckpt)
    user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])
    hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [5,10,20,30,40,50,60,70,80,90,100], user_matrix, item_matrix, rec_data.testdata.keys())
    for key in ndcg.keys():
        log.write('Topk:{:3d}, HR:{:.4f}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(key, hr[key], recall[key], ndcg[key]))
    log.close()
