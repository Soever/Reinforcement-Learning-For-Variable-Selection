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
        self.score = np.array([0.]*self.feature_num) # 每个变量的得分
        self.state = np.array([0] *self.feature_num) # 当前状态
        self.action = list(range(self.feature_num))  #当前状态可以采取的动作，1为可以采取，0不可采取

        self.acc_old = 0
    def init(self):
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
            # r2  = R2(mse,self.scalerY.var_)
            r2 = R2(mse, 1)
            acc_list.append(r2)

        return max(acc_list)


class Agent():
    def __init__(self,m,policy,reward_K=1,AOR = None):
        self.m = m
        self.action_space = list(range(m)) #[0,1,... ,m-1] m为特征维度
        self.state = np.array([0]*m) #m维数组，值为0或1，1表示该下标动作已选取
        self.AOR  = AOR
        self.policy = policy
        self.reward_K = reward_K

    def init(self):
        self.action_space = list(range(self.m))
        self.state = np.array([0] * self.m)

    def get_action(self,method,episilon=0.2):
        """
        :param method:0 : 完全随机
                      1 ：贪婪选择
                      2 ：episilon-greedy
        :return:
        """
        if method==0 :
            action = self.get_action_random() #从动作空间列表中随机选择动作
        elif method == 1 :
            action = self.get_action_greedy()
        elif method ==2 :
            random_e = np.random.uniform(0,1)
            if random_e < episilon :
                action = self.get_action_random()
            else:
                action = self.get_action_greedy()

        return action
    def get_action_random(self):
        return random.choice(self.action_space)
    def get_action_greedy(self):
        action_space_now_score = np.full_like(self.AOR, -np.inf)
        action_space_now_score[self.action_space] = self.AOR[self.action_space]  # 复制当前动作空间价值，如果动作已不在动作空间则置为负无穷
        max_value = np.max(action_space_now_score)
        max_positions = np.argwhere(action_space_now_score == max_value)
        max_positions = [item for sublist in max_positions for item in sublist]
        action = random.choice(max_positions)
        return action


    def act(self,action):
        self.state[action] = 1 #更新当前状态
        self.action_space.remove(action) #移除进行的动作
    def update_AOR(self,action,reward):
        self.AOR[action] += reward



def findbestfeatures(agent,env,iterations):
    if agent.AOR == None:
        agent.AOR =np.array([0.]*agent.m)
        for action in agent.action_space:
            agent.init()
            env.init()
            agent.act(action)
            reward = env.get_reward(agent.state)
            agent.update_AOR(action, reward)

    for i in range(iterations):
        agent.state = np.array([0]*agent.m)
        while agent.action_space :
            #agent从特征空间选择特征
            action = agent.get_action(agent.policy)
            #agent 执行动作
            agent.act(action)
            # 获得奖励
            reward = env.get_reward(agent.state)
            if reward > 0 :
                reward*=agent.reward_K
            #更新动作函数
            agent.update_AOR(action,reward)



        if i<100 :
            index  = np.argsort(agent.AOR)[::-1]
            print(' -> '.join(map(str, index+1)))



    return agent.AOR

