import h5py
import numpy as np
from PIL import Image

def pprint(name):
    print(name)

# 本地
# h5_file = r'D:\PostGraduate\pythonex\8 Graph_Based\STGA-VAD\data\frames.h5'
# 服务器
h5_file = r'/root/autodl-tmp/STGA/data/ped2_c3d.h5'
maks_file = r'/root/autodl-tmp/STGA/data/ped2_c3d.h5'


with h5py.File(h5_file, "r") as f:
    f.visit(pprint)
    print(list(f.keys()))
    """
    ['Test001.npy', 'Test002.npy', 'Test003.npy', 'Test004.npy', 'Test005.npy', 'Test006.npy', 'Test007.npy',
     'Test008.npy', 'Test009.npy', 'Test010.npy', 'Test011.npy', 'Test012.npy', 'Train001.npy', 'Train002.npy',
     'Train003.npy', 'Train004.npy', 'Train005.npy', 'Train006.npy', 'Train007.npy', 'Train008.npy', 'Train009.npy',
     'Train010.npy', 'Train011.npy', 'Train012.npy', 'Train013.npy', 'Train014.npy', 'Train015.npy', 'Train016.npy']
    """


    # 读入数据集
    dset_train = f['Train012.npy']  # <HDF5 dataset "Test002-000000": shape (16,), type "|S22905">
    print(f['Train012.npy'][:]) # 读取内容

