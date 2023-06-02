# 尝试将转化的npy文件展示其中的内容。
import numpy as np
import os
import cv2
import glob
from PIL import Image


# bmp 转换为jpg
def bmpTojpg(file_path):
    for fileName in os.listdir(file_path):
        old_path = os.path.join(file_path, fileName)
        new_path = old_path.strip().split('/')[1:-1]
        new_path = '/' + '/'.join(new_path[:4]) + '/test_frame_mask/jpg/' + '/'.join(new_path[-2:])
        # /root/autodl-tmp/STGA/data/test_frame_mask/jpg/Test/Test001_gt
        newFileName = os.path.join(fileName.split('.')[0] + ".jpg")  # 001.jpg
        im = Image.open(file_path + "/" + fileName)
        im.save(new_path + '/' + newFileName)


# jpg 转换为npy
def jpgTonpy(path):
    for fileName in os.listdir(path):
        old_path = os.path.join(path, fileName) # /root/autodl-tmp/STGA/data/test_frame_mask/jpg/Test/Test001_gt/001.jpg
        new_path = old_path.strip().split('/')[1:-1]
        new_path = '/' + '/'.join(new_path[:4]) + '/test_frame_mask/' + '/'.join(new_path[-2:])
        new_path = new_path.replace('_gt', '') # /root/autodl-tmp/STGA/data/test_frame_mask/Test/Test001
        if not
        im = Image.open(path + "/" + fileName)
        im2 = np.array(im)
        np.save(new_path, im2)


frames_dir = r'/root/autodl-tmp/STGA/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/'
jpg_dir = r'/root/autodl-tmp/STGA/data/jpg/Test'
video_list = os.listdir(frames_dir)
video_list = list(i for i in video_list if 'gt' in i)
for i in range(len(video_list)):  # 遍历 Test0xx_gt
    print('processing video {}'.format(video_list[i]))
    frames = frames_dir + video_list[i]  # 获取 Train-Train00x 下的所有 .bmp 文件
    path = '/'.join(frames.split('/')[:5]) + '/test_frame_mask/jpg/Test/' + ''.join(frames.split('/')[-1])
    # /root/autodl-tmp/STGA/data/test_frame_mask/jpg/Test/Test001_gt
    # bmpTojpg(frames)
    jpgTonpy(path)

frames_dir = r'/root/autodl-tmp/STGA/data/ped2/ped2_Test'
f = h5py.File(r'/root/autodl-tmp/STGA/data/ped2_test.h5', 'a')

video_list = os.listdir(frames_dir)  # frames_dir这个文件夹下面的所有文件（或文件夹）的名称  ['Train001', 'Test001']

for i in range(len(video_list)):  # 遍历 Train0xx/Test0xx
    print('processing video {}'.format(video_list[i]))
    frames = glob.glob(frames_dir + '/' + video_list[i] + '/*.tif')  # 获取 ped2_Train/Train00x 下的所有tif文件
    print(frames)
    feature = []
    count = 0

    for j in range(len(frames) // 16):  # 每16帧提取提取一次特征
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

    f.create_dataset(video_list[i] + '.npy', data=feature, chunks=True)

print('finished')

