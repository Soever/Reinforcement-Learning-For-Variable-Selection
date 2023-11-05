

from data_process import normalize
import numpy as np
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from model_evaluation import R2
from sklearn.metrics import r2_score
import sys
import random
import matplotlib.pyplot as plt

class Environment():
    def __init__(self, data, update_method=1, gamma=0, reward_K=1, ycol=-1):
        self.data = data
        self.y = data[:, ycol:].copy()
        self.x = data[:, :ycol].copy()
        self.scalerX = preprocessing.StandardScaler().fit(self.x)
        self.scalerY = preprocessing.StandardScaler().fit(self.y)
        self.stdX = self.scalerX.transform(self.x)
        self.stdY = self.scalerY.transform(self.y)
        self.feature_num = self.stdX.shape[1]
        self.score = np.array([0.] * self.feature_num)  # 每个变量的得分
        self.state = np.array([0] * self.feature_num)  # 当前状态
        self.action = list(range(self.feature_num))  # 当前状态可以采取的动作，1为可以采取，0不可采取

        self.acc_new = 0
        self.acc_old = 0
        self.best_R2 = 0
        self.update_method = update_method
        self.gamma = gamma
        self.best_state = None
        self.train_sample_num = self.stdX.shape[0]
        self.reward_K = reward_K

    def init(self):
        self.acc_old = 0
        self.acc_new = 0


    def get_reward(self, state, AOR):
        """
        :param state:当前状态
        :param action_t: 当前状态采取的动作
        :return: 采取当前动作后准确度得到的提升
        """

        # 计算该特征集合下的准确度
        self.acc_new = self.calculate_accuracy(state)
        if self.gamma:
            TD_target = self.calculate_Gt(state, AOR)
        else:
            TD_target = 0
        # 计算和上个状态的差值
        acc_diff = self.acc_new - self.acc_old
        if acc_diff > 0:
            acc_diff *= self.reward_K
        # 保存该状态下的准确度
        self.acc_old = self.acc_new
        return acc_diff + self.gamma * TD_target

    def calculate_accuracy(self, features_selected):
        data_train = self.stdX[:, features_selected == 1]
        acc_list = []
        for i in range(1, sum(features_selected) + 1):
            model = PLSRegression(n_components=i)
            Y_pred_std = cross_val_predict(model, data_train, self.stdY, cv=5).reshape(-1, 1)
            mse = ((Y_pred_std - self.stdY) ** 2).sum()
            r2 = R2(mse, self.train_sample_num)
            acc_list.append(r2)
        bestR2 = max(acc_list)

        if self.best_R2 < bestR2:
            self.best_R2 = bestR2
            self.best_state = features_selected
            print("最好的R2:" + str(self.best_R2))
            print("当前最优的状态为：" + str(self.best_state))

        return bestR2

    def calculate_Gt(self, features_selected, AOR):
        TD_target = 0
        if self.update_method == 1:
            indices = [i for i, val in enumerate(features_selected) if val == 0]
            if indices:
                TD_target = AOR[indices].max()

        return TD_target


class Agent():
    def __init__(self, m, policy, epsilon=0.1, pre_evaluate=True, AOR=None):
        self.m = m
        self.action_space = list(range(m))  # [0,1,... ,m-1] m为特征维度
        self.state = np.array([0] * m)  # m维数组，值为0或1，1表示该下标动作已选取
        self.AOR = AOR
        self.policy = policy
        self.epsilon = epsilon
        self.pre_evaluate = pre_evaluate
        self.visit_count = np.array([0] * m)

    def init(self):
        self.action_space = list(range(self.m))
        self.state = np.array([0] * self.m)

    def get_action(self, method):
        """
        :param method:0 : 完全随机
                      1 ：贪婪选择
                      2 ：epsilon-greedy
        :return:
        """

        if method == 0:
            action = self.get_action_random()  # 从动作空间列表中随机选择动作
        elif method == 1:
            action = self.get_action_greedy()
        elif method == 2:
            random_e = np.random.uniform(0, 1)
            if random_e < self.epsilon:
                action = self.get_action_random()
            else:
                action = self.get_action_greedy()
        else:
            action = -1
            raise ValueError("该策略未定义，请检查策略设置")

        return action

    def get_action_random(self):
        # 随机获得动作
        return random.choice(self.action_space)

    def get_action_greedy(self):
        # 复制当前动作空间价值，如果动作已不在动作空间则置为负无穷
        action_space_now_score = np.full_like(self.AOR, -np.inf)
        action_space_now_score[self.action_space] = self.AOR[self.action_space]
        # 找到当前动作空间价值最高的动作，若动作不唯一 则随机从最高价值的动作中选择一个
        max_value = np.max(action_space_now_score)  # 找动作价值最大值
        max_positions = np.argwhere(action_space_now_score == max_value).reshape(-1)  # 找到最大值对应动作
        action = random.choice(max_positions)  # 随机选择最大值中的一个
        return action

    def act(self, action):
        self.state[action] = 1  # 更新当前状态
        self.action_space.remove(action)  # 移除进行的动作

    def update_AOR(self, action, reward):
        self.visit_count[action] += 1  # 对应动作更新次数+1
        self.AOR[action] += (reward - self.AOR[action]) / self.visit_count[action]


def find_best_features(agent, env, iterations):
    # 如果特征初始值为空，则初始化一个全为0的特征评分数组，对每个特征做一个初始评估
    if agent.AOR is None:
        agent.AOR = np.array([0.] * agent.m)
        # 对于每个动作，通过增加单个特征得到的R2指标来初始化每个特征的评分
        # 原因：若不初始化，若在0->1的特征选择
        if agent.pre_evaluate:
            for action in agent.action_space:
                agent.init()
                env.init()
                agent.act(action)
                reward = env.get_reward(agent.state, agent.AOR)
                agent.update_AOR(action, reward)

    for i in range(iterations):
        agent.init()  # 初始化智能体状态
        env.init()
        # 按照策略选择特征，进行特征评估，直到所有特征都被选择
        while agent.action_space:
            action = agent.get_action(agent.policy)  # agent从特征空间选择特征  #############
            agent.act(action)  # agent 执行动作
            reward = env.get_reward(agent.state, agent.AOR)  # 获得奖励       #############
            agent.update_AOR(action, reward)  # 更新动作函数

        if i % 99 == 0:
            index = np.argsort(agent.AOR)[::-1]
            print(' -> '.join(map(str, index + 1)))
            print(np.array2string(agent.AOR, precision=5, suppress_small=True))

    r2list = []
    agent.init()  # 初始化智能体状态
    env.init()
    while agent.action_space:
        action = agent.get_action(1)  # agent从特征空间选择特征  #############
        agent.act(action)  # agent 执行动作
        r2 = env.calculate_accuracy(agent.state)  # 获得奖励       #############
        r2list.append(r2)
    r2data = np.array(r2list)
    x_values = np.arange(1, len(r2data) + 1)
    plt.scatter(x_values, r2data)
    for i, value in enumerate(r2data):
        plt.text(i+1, value, f'{value:.3f}', ha='center', va='bottom')
    plt.plot(x_values,r2data)
    plt.show()
    return agent.AOR
