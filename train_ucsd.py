import sys

import numpy as np
import torch
from torch.autograd import Variable
from utils.utils import *
from tqdm import tqdm
from utils.eval_utils import eval, cal_false_alarm
import argparse

from model import STGA

from adj_cal import *
from torch.utils.data.dataloader import DataLoader
from dataset import Train_Dataset, Test_Dataset
# from dataset_self import Train_Dataset, Test_Dataset
import torch.nn.functional as F
import h5py


def parser_arg(count):
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='C3D_RGB')   # I3D_RGB/C3D_RGB
    parser.add_argument('--dataset', type=str, default='ucsd_ped2')  # tad

    parser.add_argument('--size', type=int, default=4096)    # I3D_RGB:1024, C3D_RGB:4096  ucf_crime:2048
    parser.add_argument('--segment_len', type=int, default=16)  # 每16帧提取一次特征
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--topk', type=int,default=7)  # top 7类
    # part num should consider the average len of the video
    parser.add_argument('--part_num', type=int, default=16)
    parser.add_argument('--part_len', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=601)
    parser.add_argument('--gpu', type=str, default='0')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('-optimizer', type=str, default='adagrad')

    parser.add_argument('--dropout_rate', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=str, default='uniform', help='[random/uniform]')

    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--clip', type=float, default=4.0)  # clip？

    parser.add_argument('--lambda_1', type=float, default=0.01)
    parser.add_argument('--lambda_2', type=float, default=0)

    parser.add_argument('--machine', type=str,default='/root/autodl-tmp/STGA/data/')  # 下述几个文件所在的前缀路径

    parser.add_argument('--feature_rgb_path', type=str, default=r'ped2_c3d.h5')  # SHT_i3d_feature_rgb.h5/SHT_c3d_feature_rgb.h5
    parser.add_argument('--model_path_log', type=str, default='log/')
    parser.add_argument('--training_txt', type=str, default='ped2_Train.txt')  # SH_Train.txt
    parser.add_argument('--testing_txt', type=str, default='ped2_Test.txt')  # SHT_Test.txt
    parser.add_argument('--test_mask_dir', type=str, default='test_frame_mask/')

    parser.add_argument('--smooth_len', type=int, default=5)

    args = parser.parse_args()
    
    args.feature_rgb_path   = args.machine + args.feature_rgb_path
    args.model_path_log     = args.machine + args.model_path_log + str(count) + '/'
    args.training_txt       = args.machine + 'ped2/' + str(count) + '/' + args.training_txt
    args.testing_txt        = args.machine + 'ped2/' + str(count) + '/' + args.testing_txt
    args.test_mask_dir      = args.machine + args.test_mask_dir + str(count) + '/'


    return args


feature_type = {
    'I3D_RGB': [16, 1024],
    'C3D_RGB': [16, 4096],
}

# bceloss = torch.nn.BCELoss(reduction='mean')

def topk_rank_loss(args, y_pred):
    topk_pred = torch.mean(
        torch.topk(y_pred.view([args.batch_size*2, args.part_num*args.part_len]), args.topk, dim=-1)[0], dim=-1, keepdim=False)
    #y_pred=(80,224=32*7,1) topk_pred=(80,)

    nor_max = topk_pred[:args.batch_size]
    abn_max = topk_pred[args.batch_size:]

    # nor_loss = bceloss(nor_max, torch.zeros(args.batch_size).cuda())

    err = 0
    for i in range(args.batch_size):
        err += torch.sum(F.relu(1-abn_max+nor_max[i]))
    err = err/args.batch_size**2

    abn_pred = y_pred[args.batch_size:]
    spar_l1 = torch.mean(abn_pred)
    nor_pred = y_pred[:args.batch_size]
    smooth_l2 = torch.mean((nor_pred[:, :-1]-nor_pred[:, 1:])**2)

    loss = err+args.lambda_1*spar_l1

    return loss, err, spar_l1, smooth_l2



def train(args):

    dataset = Train_Dataset(args.part_num, args.part_len, args.feature_rgb_path, args.training_txt, args.sample, None, args.norm)
    # print(dataset) <dataset.Train_Dataset object at 0x7fae4d7c07f0>
    # dataset.__len__()) 4
    print('=====train dataset is ok=====')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    # print(r'1:{}'.format(dataloader.__sizeof__()))  4
    print('=====dataloader is ok=====')

    model = STGA(nfeat=args.size, nclass=1).cuda().train()
    # union = Union(model).cuda().train()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                 lr=0.001,
    #                 alpha=0.99,
    #                 eps=1e-08,
    #                 weight_decay=5e-4,
    #                 momentum=0,
    #                 centered=False)


    test_feats, test_labels, test_annos = Test_Dataset(args.testing_txt, args.test_mask_dir, args.feature_rgb_path, args.norm)
    # test_feats = 18个 n*4096 的ndarray
    # test_labels = 18
    # test_annos = 18  ==> 测试集共18个数据集
    print('=====test dataset is ok=====')


    best_AUC = 0
    best_far = 1
    best_AP = 0
    best_iter = 0
    best_far_inter=0
    best_AP_inter=0
    count = 0

    a1_test = []  # a1_test是邻接矩阵吗？
    for k in range(len(test_feats)):
        # test_feats 中的每一个feature的维度为 => (11, 4096)
        dim = test_feats[k].shape[0]  # 11
        a = np.zeros((dim, dim))  # a.shape => (11, 11)
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    a[i][j] = 1.0
                else:
                    a[i][j] = a[j][i] = 1.0 / abs(i - j)
        a1_test.append(a)
    # a1_test: 18 * ndarray

    a1 = np.zeros((args.batch_size*2*args.part_num*args.part_len, args.batch_size*2*args.part_num*args.part_len))
    # a1.shape => (640, 640) ? (720, 720)
    for i in range(args.batch_size*2*args.part_num*args.part_len):
        for j in range(i, args.batch_size*2*args.part_num*args.part_len):
            if i == j:
                a1[i][j] = 1.0
            else:
                a1[i][j] = a1[j][i] = 1.0 / abs(i - j)
    a1 = torch.from_numpy(a1).cuda().float()  # torch.Size([640, 640]), torch.float32

    for epoch in range(args.epochs):
        for norm_feats, norm_labs, abnorm_feats, abnorm_labs in dataloader:
            feats = torch.cat([norm_feats, abnorm_feats], dim=0).cuda().float().view([args.batch_size*2, args.part_num*args.part_len, args.size])
            # feats => torch.Size([8, 80, 4096])
            # labs = torch.cat([norm_labs, abnorm_labs], dim=0).cuda().float().view(
            #     [args.batch_size * 2, args.part_num * args.part_len, 1])

            feats = feats.view([-1, feats.shape[-1]])  # torch.Size([640, 4096])

            outputs = model(feats, a1)  # torch.Size([640, 1])

            outputs = outputs.view([args.batch_size * 2, args.part_num * args.part_len, -1])  # torch.Size([8, 80, 1])
            outputs_mean = torch.mean(outputs, dim=-1, keepdim=True)

            loss, err, l1, l3 = topk_rank_loss(args, outputs_mean)

            optimizer.zero_grad()  # 在每次迭代之前调用这个函数，避免优化器在更新权重时累加之前的梯度
            loss.backward()  # 反向传播求梯度

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()  # 更新模型中所有的可学习参数

            # if count % 10 == 0:
            print('[{}/{}]: loss {:.4f}, err {:.4f}, l1 {:.4f}, l3 {:.4f}'.format(
                count, epoch, loss, err, l1, l3))
            count += 1
        count = 0
        dataloader.dataset.shuffle_keys()

        if epoch % 5 == 0:

            total_scores = []
            total_labels = []
            total_normal_scores = []

            with torch.no_grad():
                model = model.eval()
                n = 0
                for test_feat, label, test_anno in zip(test_feats, test_labels, test_annos):
                    # test_feat(11, 4096), label='Normal', test_anno(176, 240, 360)  ??test_anno=(764,)
                    test_feat = np.array(test_feat).reshape([-1, args.size])  # (11, 4096)
                    temp_score = []

                    a11_test = torch.from_numpy(a1_test[n]).cuda().float()
                    n += 1
                    test_feat = torch.from_numpy(test_feat).cuda().float()
                    logits = model(test_feat, a11_test)  # torch.Size([11, 1])

                    for i in range(logits.shape[0]):
                        # temp_score.extend([logits[i][0].item()] * args.segment_len)  # 16 * N
                        temp_score.extend([logits[i][0].item()])
                        if label == 'Normal':
                            total_normal_scores.extend([logits[i][0].item()])

                    total_labels.extend(test_anno[:len(temp_score)].tolist())
                    total_scores.extend(temp_score)

            total_labels = [1 if element != 0 else 0 for element in total_labels]
            auc, far, AP = eval(total_scores, total_labels)

            if far < best_far:
                best_far_inter = epoch
                best_far = far

            if AP > best_AP:
                best_AP = AP
                best_AP_inter = epoch

            if auc > best_AUC:
                name = r'{}_dataset_{}_optim_{}_lr_{}_epoch_{}_AUC_{}.pth'\
                    .format(args.type,args.dataset,args.optimizer,args.lr,epoch, auc)
                if not os.path.exists(args.model_path_log):
                    os.makedirs(args.model_path_log)
                torch.save(model.state_dict(), args.model_path_log + name)

                best_iter = epoch
                best_AUC = auc

            print('best_AUC {} at epoch {}'.format(best_AUC, best_iter))
            print('current auc: {}'.format(auc))
            print('best_far {} at epoch {}'.format(best_far, best_far_inter))
            print('current far: {}'.format(far))
            print('best_AP {} at epoch {}'.format(best_AP, best_AP_inter))
            print('current AP: {}'.format(AP))
            print('===================')
            model = model.train()
            print(r'yey~{}'.format(epoch))


if __name__ == '__main__':
    # 10次训练
    for i in range(10):
        print('-' * 30)
        print(r'the is the {}th training.'.format(i))
        args = parser_arg(count=i)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        set_seeds(args.seed)
        train(args)


