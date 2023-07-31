from data_process import excel2np
from data_process import normalize
import numpy as np
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from model_evaluation import R2
import sys


class environment():
    def __init__(self,data,ycol = -1):
        self.data = data
        self.y = data[:,ycol:].copy()
        self.x = data[:,:ycol].copy()
        self.scalerX = preprocessing.StandardScaler().fit(self.x)
        self.scalerY = preprocessing.StandardScaler().fit(self.y)
        self.stdX = self.scalerX.transform(self.x)
        self.stdY = self.scalerY.transform(self.y)
        self.feature_num = self.stdX.shape[0]
        self.score = np.array([0.]*self.feature_num)
        self.state = np.array([0] *self.feature_num)
        self.action = list(range(self.feature_num))


    def get_reward(self, action_t):
        """
        :param state_t:当前状态
        :param action_t: 当前状态采取的动作
        :return: 采取当前动作后准确度得到的提升
        """
        pls = PLSRegression()
        #1 计算当前状态的准确度
        feature_t = self.state.copy()
        acc_old = self.calculate_accuracy(feature_t)
        #2.计算采取动作后的准确度
        feature_t[action_t] = 1
        acc_new = self.calculate_accuracy(feature_t)

        return acc_new-acc_old

    def calculate_accuracy(self,features_selected):
        data_train = self.stdX[features_selected]
        acc_list =[]
        for i in range(1, sum(features_selected)):
            model = PLSRegression(n_components=i)
            Y_pred = cross_val_predict(model, data_train, self.stdY, cv=5)
            mse = ((Y_pred-self.stdY)**2).sum()/len(Y_pred)
            r2  = R2(mse,self.scalerY.var_)
            acc_list.append(r2)

        return max(acc_list)