# SIGIR23-VGCL
Implementation of our SIGIR 2023 accepted paper "Generative-Contrastive Graph Learning for Recommendation".
PDF file is here: https://le-wu.com/files/Publications/CONFERENCES/SIGIR-23-yang.pdf
![](https://github.com/yimutianyang/SIGIR23-VGCL/blob/main/framework.jpg)

In this work, we investigate GCL-based recommendation from the perspective of better contrastive view construction, and propose a
novel Variational Graph Generative-Contrastive Learning (VGCL) framework. Instead of data augmentation, we leverage the variational
graph reconstruction technique to generate contrastive views to serve contrastive learning. Specifically, we first estimate each node’s
probability distribution by graph variational inference, then generate contrastive views with multiple samplings from the estimated
distribution. As such, we build a bridge between the generative and contrastive learning models for recommendation. The advantages
have twofold. First, the generated contrastive representations can well reconstruct the original graph without information distortion.
Second, the estimated variances vary from different nodes, which can adaptively regulate the scale of contrastive loss for each node.
Furthermore, considering the similarity of the estimated distributions of nodes, we propose a cluster-aware twofold contrastive
learning, a node-level to encourage consistency of a node’s contrastive views and a cluster-level to encourage consistency of nodes
in a cluster. Empirical studies on three public datasets clearly show the effectiveness of the proposed framework.

Prerequisites
-------------
* Please refer to requirements.txt

Usage-TF
-----
* python run_VGCL.py --dataset douban_book --gcn_layer 2 --alpha 0.2 --gamma 0.4 --temp_cluster 0.13 --num_user_cluster 900 --num_item_cluster 300
* python run_VGCL.py --dataset dianping --gcn_layer 3 --alpha 0.05 --gamma 0.5 --temp_cluster 0.15 --num_user_cluster 500 --num_item_cluster 100
* python run_VGCL.py --dataset ml25m --gcn_layer 3 --alpha 0.1 --gamma 1.0 --temp_cluster 0.08 --num_user_cluster 1200 --num_item_cluster 100 --batch_size 4096

Usage-Torch
------
cd torch_version
* python run_VGCL.py --dataset douban_book --gcn_layer 2 --alpha 0.2 --gamma 0.4 --temp_cluster 0.13 --num_user_cluster 900 --num_item_cluster 300
* python run_VGCL.py --dataset dianping --gcn_layer 3 --alpha 0.05 --gamma 0.5 --temp_cluster 0.15 --num_user_cluster 500 --num_item_cluster 100
* python run_VGCL.py --dataset ml25m --gcn_layer 3 --alpha 0.1 --gamma 1.0 --temp_cluster 0.08 --num_user_cluster 1200 --num_item_cluster 100 --batch_size 4096

Notice
------
* All experimental results reported in the paper are based on **TensorFlow implementation**.
* There are slight differences in Pytorch implementation. 
* Running speed also differs from platforms, VGCL runs much faster on the Tensorflow platform than Pytorch.

Citation
--------
If you find this useful for your research, please kindly cite the following paper:<br>
```
@article{VGCL2023,
  title={Generative-Contrastive Graph Learning for Recommendation},
  author={Yonghui Yang, Zhengwei Wu, Le Wu, Kun Zhang, Richang Hong, Zhiqiang Zhang, Jun Zhou and Meng Wang}
  jconference={46nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}
```

Author contact:
--------------
Email: yyh.hfut@gmail.com

