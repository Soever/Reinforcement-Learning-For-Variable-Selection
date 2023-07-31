from data_process import excel2np
from data_process import normalize
import numpy as np
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from model_evaluation import R2
import sys
import random

class Environment():
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

        self.acc_old = 0

    def get_reward(self,state):
        """
        :param state:当前状态
        :param action_t: 当前状态采取的动作
        :return: 采取当前动作后准确度得到的提升
        """


        #计算该特征集合下的准确度
        acc_new = self.calculate_accuracy(state)
        #计算和上个状态的差值
        acc_diff = acc_new-self.acc_old
        #保存该状态下的准确度
        self.acc_old = acc_new
        return acc_diff

    def calculate_accuracy(self,features_selected):
        data_train = self.stdX[:,features_selected==1]
        acc_list =[]
        for i in range(1, sum(features_selected)+1):
            model = PLSRegression(n_components=i)
            Y_pred = cross_val_predict(model, data_train, self.stdY, cv=5)
            mse = ((Y_pred-self.stdY)**2).sum()
            r2  = R2(mse,self.scalerY.var_)
            acc_list.append(r2)

        return max(acc_list)


class Agent():
    def __init__(self,n):
        self.n = n
        self.action_space = list(range(n))
        self.state = np.array([0]*n)

    def get_action(self):
        action = random.choice(self.action_space)
        self.state[action] = 1
        self.action_space.remove(action)
        return action




def findbestfeatures(agent,env):
    AOR = np.array([0.]*agent.n)
    for i in range(100000000):
        agent.state = np.array([0]*agent.n)
        while agent.action_space :
            #agent从特征空间选择特征
            action = agent.get_action()
            reward = env.get_reward(agent.state)
            AOR[action] += reward

    return AOR

