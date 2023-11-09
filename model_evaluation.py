import math
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from data_process import normalize

def R2(a, b):
    '''
    :param a: RSS预测残差或MSE
    :param b: TSS总平方差或VAR(y)
    :return:  1-MSE/VAR 或 1-RSS/TSS
    对于标准化数据，r2 = 1-mse
    '''
    return 1 - a / b


def calculate_best_R2(X, Y,p=None):
    """
    :param X:所有数据集X
    :param Y:所有数据集Y
    :return: Q2最大值及最佳主元个数
    """
    stdX, stdY, mean, std = normalize(X, Y)
    # 从1个主成分开始记录
    N = len(stdY)
    R2_list = []
    for i in range(1, X.shape[1] + 1):
        # 建立i个主元的PLS模型
        model = PLSRegression(n_components=i,scale=False)
        # 交叉验证
        stdY_pred = cross_val_predict(model, stdX, stdY)
        # 计算R2
        mse = np.sum((stdY_pred.reshape(stdY.shape) - stdY)**2) / N
        R2_value = 1-mse
        # 记录R2值
        R2_list.append(R2_value)
    # 求最佳FPE
    bestvalue = max(R2_list)
    return bestvalue, R2_list.index(bestvalue)+1