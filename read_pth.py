"""
    .pth文件里保存的是训练好的模型的参数，比如：权值（weight），偏置（bias）等
"""

import torch

net = torch.load('../data/c3d_pretrained.pth')
print(net)
print(type(net))
print(len(net))
for k in net.keys():
    print(k)
# for key, value in net["model"].items():
#     print(key, value.size(), sep=" ")

