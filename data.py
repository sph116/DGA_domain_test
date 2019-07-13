import tldextract
import pandas as pd
import re
import pickle
import numpy as np


"""
提取100万360_DGA 域名作 与100万top-1m域名 作为训练数据
提取特征与替换目标值 构造为[(目标值, 特征值), ()....] 的数据集
将重要参数 与 数据集 保存为 pki文件方便调用
"""

def get_alexa():
    """
    提取alexa 域名 全部为正面域名 结构为 [[域名1, 目标值], [域名2, 目标值]..........]
    :return:
    """
    alexa_date = pd.read_csv("./raw_data/top-1m.csv").iloc[0:1000000, :]
    return [("benign", tldextract.extract(row["num1"]).domain) for index, row in alexa_date.iterrows()]

def get_360_DGA():
    """
    提取360dga域名 结构同上
    :return:
    """
    f = open("./raw_data/360_dga.txt", "r", encoding="utf-8").readlines()[0: 1000000]
    # ls = [re.sub('\t+', ' ', i).split(' ')[0] for i in f]
    # d = sorted(Counter(ls).items(), key=lambda x: x[1], reverse=True)[0: 100]
    # print(d)
    return [(re.sub('\t+', ' ', i).split(' ')[0], tldextract.extract(re.sub('\t+', ' ', i).split(' ')[1]).domain) for i in f]



# print(len(get_360_DGA()))
def get_zeus_dga():
    """
    提取zeus域名
    :return:
    """
    f = open("zeus_dga_domains.txt", "r", encoding="utf-8").read()
    return f.split(".")


def get_data():
    """
    拼接返回全部数据 作为数据集
    :return:
    """
    return get_alexa() + get_360_DGA()

date_set = get_data()
features = [i[1] for i in date_set]   #提取域名
label = [i[0] for i in date_set]      #提取标签

valid_chars = {x:idx+1 for idx, x in enumerate(set(''.join(features)))} #构造检索字典
max_features = len(valid_chars) + 1  #特征维度 检索字典的长度
maxlen = np.max([len(x) for x in features])   #特征数量

item = {"date_set": date_set, "valid_chars": valid_chars, "max_features": max_features, "maxlen": maxlen}

#存储数据
fp = open('./data_set/all_dataV1.pkl', 'wb')
pickle.dump(item, fp)
fp = open("./data_set/all_dataV1.pkl", 'rb')
print(len(pickle.load(fp)))