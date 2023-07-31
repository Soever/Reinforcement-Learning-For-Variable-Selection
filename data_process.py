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

def importData(xDataPath, yDataPath):
    # 读取数据集X，Y
    dfx = pd.read_csv(xDataPath, low_memory=False)
    dfy = pd.read_csv(yDataPath, low_memory=False)

    # 读取X采样时间
    timex = dfx.loc[0, "Time"]

    # 删除X，Y前两行
    dfx.drop(dfx.head(2).index, inplace=True)
    dfx = dfx.reset_index(drop=True)
    dfy.drop(dfx.head(2).index, inplace=True)
    dfy = dfy.reset_index(drop=True)

    # 将Time转为时间戳
    dfx['Time'] = pd.to_datetime(dfx['Time'])
    dfy['Time'] = pd.to_datetime(dfy['Time'])
    # 将Time设置为索引
    dfx = dfx.set_index('Time')
    dfy = dfy.set_index('Time')
    # 将y采样变成30s，其余时间变成NAN
    dfy = dfy.resample('30S').asfreq()
    # 按照时间戳合并X，Y
    df = pd.merge(left=dfx, right=dfy, how='outer', on='Time')

    # 除掉包含任何包含Nan的行
    df = df.dropna(axis=0, how='any')

    # 重置索引为默认索引
    # df = df.reset_index()
    return df