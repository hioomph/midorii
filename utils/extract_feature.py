import glob
import cv2
import h5py
import os
# from opencv_transforms import transforms
import torch
from opencv_videovision.transforms import *
from tqdm import tqdm
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable



mean = [105.80, 100.25, 99.08]
std = [65.88, 61.44, 63.85]

video_transform_list = [Resize([256, 340]), CenterCrop(224), ClipToTensor(div_255=False)]
video_transform = Compose(video_transform_list)


class C3DBackbone(nn.Module):
    def __init__(self):
        super(C3DBackbone, self).__init__()
        # 112
        self.conv1a = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # 56
        self.conv2a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 28
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 14
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 7
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        # self.pool5 = nn.AdaptiveAvgPool3d(1)
        self.fc6 = nn.Linear(8192, 4096)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.pool1(x)
        x = self.relu(self.conv2a(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        out_4 = x

        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        return x


net = C3DBackbone()
# state_dict = torch.load('D:/code/CHY/GCR_VAD/data/C3D_Sport1M.pth')
#
# new_dict = {}
# for key, value in state_dict.items():
#     # new_dict[key] = value
#     if key[9:] != 'fc7.weight' and key[9:] != 'fc7.bias' and key[9:] != 'fc8.weight' and key[9:] != 'fc8.bias':
#         new_dict[key[9:]] = value
#
# net.load_state_dict(new_dict)

# 本机
# checkpoint = torch.load('../data/c3d_pretrained.pth')
# 服务器
checkpoint = torch.load('/root/autodl-tmp/STGA/data/c3d_pretrained.pth')

state_dict = net.state_dict()
base_dict = {}
checkpoint_keys = checkpoint.keys()
for k, v in state_dict.items():
    for _k in checkpoint_keys:
        if k in _k:
            base_dict[k] = checkpoint[_k]
state_dict.update(base_dict)
net.load_state_dict(state_dict)

net.cuda(0)
net.eval()

# 本机
# frames_dir = r'../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001'
# # output_dir = 'D:/code/CHY/anomaly_datasets/Avenue/C3D_feature'
# f = h5py.File('../data/frames.h5', 'a')

# 服务器
frames_dir = r'/root/autodl-tmp/STGA/data/ped2/'
f = h5py.File(r'/root/autodl-tmp/STGA/data/ped2_c3d.h5', 'a')

video_lists = os.listdir(frames_dir)  # 所有帧

for m in range(len(video_lists)):
    video_list = os.listdir(os.path.join(frames_dir, video_lists[m]))
    for i in range(len(video_list)):  # 遍历 Train0xx/Test0xx
        print('processing video {}'.format(video_list[i]))
        frames = glob.glob(frames_dir + video_lists[m] + '/' + video_list[i] + '/*.tif')  # 获取 ped2_Train/Train00x 下的所有tif文件
        print(frames)
        feature = []
        count = 0

        for j in range(len(frames)//16):  # 每16帧提取提取一次特征
            temp_frames = []
            for k in range(16):
                # frame = Image.open(frames[count])
                # frame = np.array(frame)  # 每一帧即为一张tif文件
                img = cv2.cvtColor(cv2.imread(frames[count], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)  # BGR转RGB格式
                frame = np.array(img)  # img.shape ==> (158, 238, 3)
                temp_frames.append(frame)
                count = count + 1

            # temp_frames = transforms(temp_frames)
            temp_frames = video_transform(temp_frames)  # temp_frames.shape ==> torch.Size([16, 224, 224])
            temp_frames = Variable(temp_frames).cuda()
            temp_frames = temp_frames.reshape(1, 3, 16, 224, 224)

            temp_feature = net(temp_frames)
            temp_feature = torch.mean(temp_feature, dim=0)
            temp_feature = temp_feature.cpu().detach().numpy()

            feature.append(temp_feature)

        print(video_list[i] + '.npy')
        f.create_dataset(video_list[i] + '.npy', data=feature, chunks=True)

print('finished')
