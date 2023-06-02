import h5py
import numpy as np

def read_matlab_file(file_path):
    data_list = []  # 用于存储读取的数据

    # 打开 MATLAB 文件
    with h5py.File(file_path, 'r') as file:
        # 递归读取文件中的所有数据集
        def read_dataset(name, dataset):
            if isinstance(dataset, h5py.Dataset):
                data_list.append(len(dataset[()]))


        file.visititems(read_dataset)

    return data_list[1:-3]

# 读取 MATLAB 文件
# matlab_file_path = '/root/autodl-tmp/STGA/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/UCSDped2.mat'
# data = read_matlab_file(matlab_file_path)
# print(data)
