import pandas as pd
import numpy as np

def excel2np(path):
    df = pd.read_excel(path, header=[0, 1])
    # 转换为 Numpy 数组
    array = df.values[:,1:]
    X = array[:,:-3]
    Y = array[:,-3:]
    return X,Y

def normalize(X, Y):
    """
    :param X: 原始数据X
    :param Y: 原始数据Y
    :return: 标准化数据x，标准化数据y，均值列表，标准差列表
    """
    # 初始化均值、标准差列表、标准X、标准Y
    mean = []
    STD = []
    stdX = np.empty_like(X)
    stdY = np.empty_like(Y)
    # 每列标准化
    for i in range(X.shape[1]):
        # 求均值mean
        m = np.mean(X[:, i])
        # 求方差var
        v = np.var(X[:, i])
        # 求标准差std
        v = v ** 0.5
        mean.append(m)
        STD.append(v)
        # 标准化 X =  (x-mean)/std
        stdX[:, i] = (X[:, i] - m) / v
    # 求均值mean
    m = np.mean(Y)
    # 求标准差std
    v = np.var(Y)
    v = v ** 0.5
    mean.append(m)
    STD.append(v)
    # 标准化Y
    stdY = (Y - m) / v

    return stdX, stdY, mean, STD