import numpy as np
import scipy.misc

path = '/root/autodl-tmp/STGA/data/test_frame_mask/Test001.npy'
data = np.load(path)

# print('type :', type(data))
# print('shape :', data.shape)
# print('data :')
# print(data)

imgs_test = np.load(path)  # 读入.npy文件
print(imgs_test.shape)  # (240, 360)
for i in range (imgs_test.shape[0]):
    B = imgs_test[i, 0, : ,:]#对图像维度进行改变
    scipy.misc.imsave("./" + str(i) + "_predResults" + ".jpg",B)  # 保存为png格式，也可将png换位jpg等其他格式



