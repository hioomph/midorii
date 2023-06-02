"""
    对于UCSD_Ped2的训练集进行划分：
        在原训练集中随机选取4个，原测试集中随机选取6个组成新的训练集，剩余的组成新的测试集。
        重复上述过程，进行10次训练，对最后的结果取均值。
"""
import os
import random
import shutil

# 原训练集所在地址
source_train_path = '/root/autodl-tmp/STGA/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
source_test_path = '/root/autodl-tmp/STGA/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
# 新训练集中随机挑选的 原训练集/原测试集 数目
train_selected = 4
test_selected = 6
# 新的目标路径
target_new_path = '/root/autodl-tmp/STGA/data/ped2/'


def check_common_characters(folder_name1, folder_name2):
    common_chars = set(folder_name1) & set(folder_name2)
    return len(common_chars) == 7

def make_txt(target_new_path, new_dataset, pre1, count):
    """
        txt文件构成： Train0xx/Test0xx,标签,总帧数
    """
    for i in range(len(new_dataset)):
        pre2 = str(new_dataset[i][:-3])
        path = os.path.join(target_new_path, str(count), 'ped2_' + pre1 + '.txt')
        content = [new_dataset[i]]  # 要写入txt文件的内容

        if pre2 == 'Train': label = 0
        else: label = 1
        content.append(str(label))
        len_sum = len(os.listdir(os.path.join('/root/autodl-tmp/STGA/data/UCSD_Anomaly_Dataset.v1p2/UCSDped2',
                                              pre2, new_dataset[i])))
        content.append(str(len_sum))
        content = ','.join(content)

        # 打开文件并写入内容
        with open(path, 'a+') as file:
            file.seek(0)  # 将文件指针移到文件开头
            data = file.read()
            if len(data) > 0:
                file.write('\n')  # 若文件非空，则在文件末尾先写入换行符
            file.write(content)  # 写入新行

    # 打印成功消息
    print(r'{}_{}.txt is ok!'.format(count, pre1))

def random_select_train_folders(source_path, target_folder_path, num_folders, str_need):
    count = 0
    for i in range(10):
        # 获取源文件夹路径集合
        folder_list = os.listdir(source_path)

        # 筛选出含有 "Train" 或 "Test" 字符的文件夹
        train_folders = [folder for folder in folder_list if (str_need in folder) and ('gt' not in folder)]

        # 随机选择 num_folders 个文件夹
        selected_folders = random.sample(train_folders, num_folders)

        # 新建本次循环的新路径
        target_folder_path = '/root/autodl-tmp/STGA/data/ped2/' + str(count)
        target_path1 = os.path.join(target_folder_path, 'ped2_Train')
        target_path2 = os.path.join(target_folder_path, 'ped2_Test')

        new_train = []  # 新训练集集合
        new_test = []  # 新测试集集合

        # 复制选中的文件夹到目标路径1（新训练集路径），剩下的文件夹复制到目标路径2（新测试集路径）
        for folder in train_folders:
            source_folder_path = os.path.join(source_path, folder)
            if folder in selected_folders:
                target_folder_path = os.path.join(target_path1, folder)
                new_train.append(folder)
            else:
                target_folder_path = os.path.join(target_path2, folder)
                new_test.append(folder)
            shutil.copytree(source_folder_path, target_folder_path)

        # 'gt' 文件夹存储到单独的文件夹
        gt_folders = [folder for folder in folder_list if (str_need in folder) and ('gt' in folder)]
        for folder in gt_folders:
            source_folder_path = os.path.join(source_path, folder)
            path = os.path.join(target_new_path, str(count), 'gt')
            target_folder_path = os.path.join(path, folder)
            shutil.copytree(source_folder_path, target_folder_path)


        make_txt(target_new_path, new_train, 'Train', count)
        make_txt(target_new_path, new_test, 'Test', count)

        count += 1


random_select_train_folders(source_train_path, target_new_path, train_selected, "Train")
random_select_train_folders(source_test_path, target_new_path, test_selected, "Test")
