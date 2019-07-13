# from data import get_data
# import numpy as np
# from keras.preprocessing import sequence
# from keras.models import load_model
# import pickle
# from random import shuffle
#
# def get_features():
#     fp = open("./data_set/all_data.pkl", 'rb')
#     date_set = pickle.load(fp)
#     shuffle(date_set)
#     features = [i[1] for i in date_set]  # 提取域名
#     label = [i[0] for i in date_set]  # 提取标签
#     return features, label
# def ready_data():
#
#     features, label = get_features()
#     valid_chars = {x: idx + 1 for idx, x in enumerate(set(''.join(features)))}  # 构造检索字典
#     maxlen = np.max([len(x) for x in features])  # 获得输入特征的最大长度
#     return valid_chars, maxlen
#
# def process_Y(label):
#     return [0 if x == 'benign' else 1 for x in label]
#
# def process_X(features):
#     valid_chars, maxlen = ready_data()
#     print(valid_chars, maxlen)
#     X = [[valid_chars[y] for y in x] for x in features]
#     X = sequence.pad_sequences(X, maxlen=maxlen)
#     X = np.array(X)
#     return X
#
# def DGA_prodect():
#     x, y = get_features()
#     X = process_X(x[0: 50000])
#     Y = process_Y(y[0: 50000])
#     model = load_model('./model/DGA_predict_LSTM_V1.h5')
#     score, acc = model.evaluate(X, Y, batch_size=128)
#     print(score, acc)
#     prodect_label = model.predict(X)
#     # for i, j in zip(Y, prodect_label):
#     #     if j < 0.5:
#     #         print(0, j, i)
#     #     elif j > 0.5:
#     #         print(1, j, i)
#
#
# DGA_prodect()

import matplotlib.pyplot as plt
import time
from keras.preprocessing import sequence
import pickle
from sklearn.utils import shuffle
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import random
from random import shuffle
import tldextract
random.seed(1024)


"""
提取数据集 与参数 训练神经网络 
处理数据 以及预测
预测数值 以0.5为分界   大于0.5与小于0.5分为二类
"""

fp = open("./data_set/all_data.pkl", 'rb')
all_data = pickle.load(fp)  #提取数据

valid_chars = all_data["valid_chars"]
print(valid_chars)


max_features = all_data["max_features"]         #特征维度
maxlen = all_data["maxlen"]     #每条训练数据的特征数量


def process_features(valid_chars, features, maxlen):
    """将特征替换为 检索字典的值 并根据最长特征长度 构造每个特征 无值填充为0"""
    X = [[valid_chars[y] for y in x] for x in features]   #域名 根据检索字典 转换为 数字特征
    X = sequence.pad_sequences(X, maxlen=maxlen)    #根据元素最大长度
    X = np.array(X)   #转换为nd.array
    return X




def show_train_history(train_history, train, velidation):
    """
    可视化训练过程 对比
    :param train_history:
    :param train:
    :param velidation:
    :return:
    """
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[velidation])
    plt.title("Train History")  # 标题
    plt.xlabel('Epoch')  # x轴标题
    plt.ylabel(train)  # y轴标题
    plt.legend(['train', 'test'], loc='upper left')  # 图例 左上角
    plt.show()


def build_model(max_features, maxlen):
    """ 定义模型 """
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=128, input_length=maxlen))
    model.add(LSTM(128))
    model.add(Dropout(0.5))   #防止过拟合
    model.add(Dense(1))      #以0.5为分界 分类
    model.add(Activation('sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def run():
    """
    训练函数
    :return:
    """
    model = build_model(max_features, maxlen)

    data_set = all_data["date_set"]
    shuffle(data_set)  # 打乱数据集 防止出现000000011111111情况
    features = [i[1] for i in data_set]  # 提取域名
    label = [i[0] for i in data_set]  # 提取标签
    y = [0 if x == 'benign' else 1 for x in label]  # 目标值修改为0或1
    x = process_features(valid_chars, features, maxlen)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)  # 数据集切割

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=3, batch_size=128, shuffle=True, validation_split=0.1)      # 训练
    end_time = time.time()
    print("花费时间为{}".format(end_time - start_time))

    model.save('./model/DGA_predict_LSTM_V4.h5')

    score, acc = model.evaluate(x_test, y_test, batch_size=128)
    print('评分:', score)
    print('测试集准确率:', acc)

    """可视化训练过程"""
    show_train_history(history, 'acc', 'val_acc')  # 训练集准确率与验证集准确率 折线图
    show_train_history(history, 'loss', 'val_loss')  # 训练集误差率与验证集误差率 折线图

    # predect_label = model.predict(x_test)
    # for (a, b, c) in zip(y_test[0: 100], x_test[0: 100], predect_label[0: 100]):
    #     print("---原域名为：", b)
    #     print("预测倾向为", a)
    #     print("真实倾向为", c)


# run()  #重新训练时 启动

def predict(data):
    """
    预测函数  data为待预测数据 输入格式为 [域名1, 域名2, 域名3......]
    :param data:
    :return:
    """
    #data = [tldextract.extract(i).domain for i in data]  #提取域名
    features = process_features(valid_chars, data, maxlen)   #提取特征
    model = load_model('./model/DGA_predict_LSTM_V4.h5')
    start_time = time.time()
    for i in range(0, 50000, 10):
        time.sleep(2)
        prodect_labels = model.predict(features[i: i+100])
        print(prodect_labels)
    # for i, j in zip(data, prodect_labels):
    #     print("原域名为:{}".format(i))
    #     print("预测倾向为:{}".format(j))

    end_time = time.time()
    print("花费时间为{}".format(end_time - start_time))
    # return prodect_labels


data = [i[1] for i in all_data["date_set"]][0: 50000]
predict(data)











