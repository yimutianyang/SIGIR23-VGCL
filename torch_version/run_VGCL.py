import os, pdb, sys
import warnings
warnings.filterwarnings("default")
import torch
import numpy as np
import argparse
from shutil import copyfile
from time import time
from tqdm import tqdm
from set import *
from VGCL import VGCL
from rec_dataset import CF_Dataset
sys.path.append('../')
from evaluate import *
from log import Logger


def parse_args():
    parser = argparse.ArgumentParser(description='VGCL Parameters')
    ### general parameters ###
    parser.add_argument('--dataset', type=str, default='douban_book', help='?')
    parser.add_argument('--runid', type=str, default='record_names', help='current log id')
    parser.add_argument('--device_id', type=str, default='0', help='?')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--topk', type=int, default=20, help='Topk value for evaluation')   # NDCG@20 as convergency metric
    parser.add_argument('--early_stops', type=int, default=10, help='model convergent when NDCG@20 not increase for x epochs')
    parser.add_argument('--num_neg', type=int, default=1, help='number of negetiva samples for each [u,i] pair')
    parser.add_argument('--social_noise_ratio', type=float, default=0, help='?')

    ### model parameters ###
    parser.add_argument('--gcn_layer', type=int, default=3, help='?')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent embedding dimension')
    parser.add_argument('--init_type', type=str, default='norm', help='?')
    parser.add_argument('--l2_reg', type=float, default=1e-4, help='?')
    parser.add_argument('--alpha', type=float, default=0.2, help='contrastive learning loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='KL term weight for ELBO based loss')
    parser.add_argument('--gamma', type=float, default=0.4, help='cluster-level contrastive loss weight')
    parser.add_argument('--temp_node', type=float, default=0.2, help='temperature for node-level contrastive learning')
    parser.add_argument('--temp_cluster', type=float, default=0.13,
                        help='temperature for cluster-aware contrastive learning')
    parser.add_argument('--num_user_cluster', type=int, default=900, help='number of user clusterings')
    parser.add_argument('--num_item_cluster', type=int, default=300, help='number of item clusterings')
    return parser.parse_args()


def eval_test(model):
    model.eval()
    with torch.no_grad():
        mean, std, embeddings_view1, _ = model.forward()
        # user_emb, item_emb = torch.split(mean, [args.num_user, args.num_item], dim=0)
        user_emb, item_emb = torch.split(embeddings_view1, [args.num_user, args.num_item], dim=0)
    return user_emb.cpu().detach().numpy(), item_emb.cpu().detach().numpy()


def makir_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_file(save_path):
    copyfile('./VGCL.py', save_path + 'VGCL.py')
    copyfile('./run_VGCL.py', save_path + 'run_VGCL.py')
    copyfile('./rec_dataset.py', save_path + 'rec_dataset.py')


if __name__ == '__main__':
    seed_everything(2023)
    args = parse_args()
    args.data_path = '../datasets/' + args.dataset + '_data/'
    record_path = '../saved/' + args.dataset + '/VGCL/' + args.runid + '/'
    model_save_path = record_path + 'models/'
    makir_dir(model_save_path)
    save_file(record_path)
    log = Logger(record_path)
    for arg in vars(args):
        log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    rec_data = CF_Dataset(args)
    args.num_user, args.num_item = rec_data.num_user, rec_data.num_item

    rec_model = VGCL(args, rec_data)
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    rec_model.to(device)
    optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)

    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    max_hr, max_recall, max_ndcg, early_stop = 0, 0, 0, 0
    topk = args.topk
    best_epoch = 0

    model_files = []
    max_to_keep = 5

    for epoch in tqdm(range(args.epochs), desc=set_color(f"Train:", 'pink'), colour='yellow',
                      dynamic_ncols=True, position=0):
        t1 = time()
        sum_auc, all_rank_loss, all_kl_loss, all_node_loss, all_cluster_loss, batch_num = 0, 0, 0, 0, 0, 0
        rec_model.train()
        #  batch数据
        user_2cluster, item_2cluster = rec_model.cluster_step()
        loader = rec_data._batch_sampling(num_negative=args.num_neg)
        for u, i, j in tqdm(loader, desc='All_batch'):
            u = torch.tensor(u).type(torch.long).to(device)  # [batch_size]
            i = torch.tensor(i).type(torch.long).to(device)  # [batch_size]
            j = torch.tensor(j).type(torch.long).to(device)  # [batch_size]
            auc, rank_loss, kl_loss, cl_loss_node, cl_loss_cluster, total_loss = \
                rec_model.calculate_all_loss(u, i, j, user_2cluster, item_2cluster)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            sum_auc += auc.item()
            all_rank_loss += rank_loss.item()
            all_kl_loss += kl_loss.item()
            all_node_loss += cl_loss_node.item()
            all_cluster_loss += cl_loss_cluster.item()
            batch_num += 1
            # pdb.set_trace()
        mean_auc = sum_auc / batch_num
        mean_rank_loss = all_rank_loss / batch_num
        mean_kl_loss = all_kl_loss / batch_num
        mean_node_loss = all_node_loss / batch_num
        mean_cluster_loss = all_cluster_loss / batch_num
        log.write(set_color('Epoch:{:d}, Train_AUC:{:.4f}, Loss_rank:{:.4f}, Loss_kl:{:.4f}, Loss_node:{:.4f}, Loss_cluster:{:.4f}\n'
                            .format(epoch, mean_auc, mean_rank_loss, mean_kl_loss, mean_node_loss, mean_cluster_loss), 'blue'))
        t2 = time()


        # ***************************  evaluation on Top-20  *****************************#
        if epoch % 1 == 0:
            early_stop += 1
            user_emb, item_emb = eval_test(rec_model)
            hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata, [20], user_emb, item_emb,
                                        rec_data.testdata.keys())
            if ndcg[20] >= max_ndcg or ndcg[20] == max_ndcg and recall[20] >= max_recall:
                best_epoch = epoch
                max_hr = hr[20]
                max_recall = recall[20]
                max_ndcg = ndcg[20]
            log.write(set_color(
                'Current Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.4f}, ndcg:{:.4f}\n'.format(epoch, topk,
                                                                                                  recall[20], ndcg[20]),
                'green'))
            log.write(set_color(
                'Best Evaluation: Epoch:{:d},  topk:{:d}, recall:{:.4f}, ndcg:{:.4f}\n'.format(best_epoch, topk,
                                                                                               max_recall, max_ndcg),
                'red'))

            if ndcg[20] == max_ndcg:
                early_stop = 0
                best_ckpt = 'epoch_' + str(epoch) + '_ndcg_' + str(ndcg[20]) + '.ckpt'
                filepath = model_save_path + best_ckpt
                torch.save(rec_model.state_dict(), filepath)
                print(f"Saved model to {filepath}")
                model_files.append(filepath)
                if len(model_files) > max_to_keep:
                    oldest_file = model_files.pop(0)
                    os.remove(oldest_file)
                    print(f"Removed old model file: {oldest_file}")

            t3 = time()
            log.write('traintime:{:.4f}, valtime:{:.4f}\n\n'.format(t2 - t1, t3 - t2))
            if epoch > 20 and early_stop > args.early_stops:
                log.write('early stop: ' + str(epoch) + '\n')
                log.write(set_color('max_recall@20=:{:.4f}, max_ndcg@20=:{:.4f}\n'.format(max_recall, max_ndcg), 'green'))
                break

    # ***********************************  start evaluate testdata   ********************************#
    rec_model.load_state_dict(torch.load(model_save_path + best_ckpt))
    user_emb, item_emb = eval_test(rec_model)
    hr, recall, ndcg = num_faiss_evaluate(rec_data.testdata, rec_data.traindata,
                                          [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], user_emb, item_emb,
                                          rec_data.testdata.keys())
    for key in ndcg.keys():
        log.write(set_color(
            'Topk:{:3d}, HR:{:.4f}, Recall:{:.4f}, NDCG:{:.4f}\n'.format(key, hr[key], recall[key], ndcg[key]), 'cyan'))
    log.close()
    print('END')