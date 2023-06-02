import glob
import os

import cv2
from PIL import Image
import numpy as np
from opencv_videovision.transforms import *

video_transform_list = [Resize([256, 340]), CenterCrop(224), ClipToTensor(div_255=False)]
video_transform = Compose(video_transform_list)

# gt帧所在地址
frames_dir_pre = '/root/autodl-tmp/STGA/data/ped2/'

for count in range(10):
    frames_dir = frames_dir_pre + str(count) + '/gt/'
    video_lists = os.listdir(frames_dir)  # Test001_gt
    mask_dir = '/root/autodl-tmp/STGA/data/test_frame_mask/' + str(count) + '/'
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    for i in range(len(video_lists)):
        frames = glob.glob(frames_dir + video_lists[i] + '/*.bmp')  # 获取当前所有 .bmp 文件
        temp = []
        count = 0

        for j in range(len(frames) // 16):  # 每16帧提取提取一次特征
            print(r'1:{}'.format(frames[count]))
            temp_frames = []
            for k in range(16):
                # 读取BMP图像
                image = cv2.cvtColor(cv2.imread(frames[count], cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)  # BGR转RGB格式
                # 将图像转换为NumPy数组
                pixel_values = np.array(image)
                temp_frames.append(pixel_values)
                count += 1

            temp_frames = video_transform(temp_frames)  # temp_frames.shape ==> torch.Size([16, 224, 224])
            temp_frames = temp_frames.reshape(1, 3, 16, 224, 224)

            temp.append(pixel_values)

        temp = np.array(temp)
        # 保存为npy文件
        video_lists[i] = ''.join(list(video_lists[i][:-3]))
        output_path = os.path.join(mask_dir, video_lists[i] + '.npy')
        np.save(output_path, temp)

print('finish!')
