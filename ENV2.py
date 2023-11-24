import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from data_process import varSelection, dataShift, dataFilter,filter_column
from model_evaluation import R2, calculate_best_R2,calculate_best_R2_Parrel
import random
import logging

from joblib import Parallel, delayed


class FSEnv():
    def __init__(self, df_class, state_size, action_size,invalid_action_reward=0, min_score=0,max_stop_step=None):
        self.df_class = df_class
        self.invalid_action_reward = invalid_action_reward
        self.min_score = min_score
        self.feature_num = self.df_class.feature_num
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.array([0] * self.feature_num * 3)  # 当前状态
        self.action = np.array([0] * self.feature_num * 3 * 2)  # 当前状态可以采取的动作，1为可以采取，0不可采取
        self.max_stop_step = self.state_size if max_stop_step is None else max_stop_step
        self.acc_new = 0
        self.acc_old = 0
        self.best_R2 = 0

        self.best_state = None

        self.lb = [0] * self.feature_num + [0] * self.feature_num + [0] * self.feature_num  # 决策变量下界
        self.ub = [60] * self.feature_num + [60] * self.feature_num + [1] * self.feature_num  # 决策变量上界


        self.done_count = 0

        #记录上一状态的数据

        self.state_last = None

        self.dataXY_history = None
        self.dataX_history = None
        self.dataY_history = None





    def init(self):
        self.acc_old = 0
        self.acc_new = 0

    def get_reward(self, state,action):
        """
        :param state:当前状态
        :param action_t: 当前状态采取的动作
        :return: 采取当前动作后准确度得到的提升
        """
        # 计算该特征集合下的准确度
        self.acc_new = self.calculate_accuracy(state,action)

        # 计算和上个状态的差值
        acc_diff = self.acc_new - self.acc_old
        if acc_diff < 0.001:
            self.done_count += 1
        else:
            self.done_count = 0
        # 保存该状态下的准确度
        self.acc_old = self.acc_new
        return acc_diff
    def calculate_accuracy(self, state,action):
        filtervalue = state[:self.feature_num]
        delayvalue = state[self.feature_num:2 * self.feature_num]
        varSelect = state[2 * self.feature_num:].astype(bool)
        originDF = self.df_class.orign_df
        bestR2 = 0
        if self.dataX_history is not None:
                feature_index = (action // 2) % self.feature_num
                if action < 4 * self.feature_num:  # t或tao 的动作
                    # 创建一个只有该动作对应特征的mask
                    single_varSelect = np.zeros(self.feature_num, dtype=bool)
                    single_varSelect[feature_index] = True
                    # 获取该特征的dataframe
                    df = varSelection(originDF, single_varSelect)
                    # 处理滞后
                    df = dataShift(df, delayvalue[single_varSelect])
                    # 处理滤波
                    single_X, y = dataFilter(df, filtervalue[single_varSelect])
                    # 更新历史data
                    self.dataX_history[:,feature_index]=single_X.reshape(-1)
                    X = self.dataX_history[:, varSelect]
                    bestR2, latent_var_num = calculate_best_R2_Parrel(X, y)
                else:  # c 的动作
                    if varSelect.sum() > 0:
                        X = self.dataX_history[:,varSelect]
                        y = self.dataY_history
                        bestR2, latent_var_num = calculate_best_R2_Parrel(X, y)
        else:
            p = varSelect.sum()
            if p > 0:
                # 选择变量
                df = varSelection(originDF, varSelect)
                # 处理滞后
                df = dataShift(df, delayvalue[varSelect])
                # 一阶滤波
                X, y = dataFilter(df, filtervalue[varSelect])
                bestR2, latent_var_num = calculate_best_R2_Parrel(X, y)

        if self.best_R2 < bestR2:
            self.best_R2 = bestR2
            self.best_state = state
            logging.debug(f"bestR2: {self.best_R2},best_state: {self.best_state}")
            # print("最好的R2:" + str(self.best_R2))
            # print("当前最优的状态为：" + str(self.best_state))
        return bestR2


    def reset(self):
        """
        重置环境到初始状态
        """
        self.state = self.init_state_to_zero()
        self.state_last = self.state.copy()
        self.acc_old = 0
        self.best_R2 = 0
        self.best_state = None
        self.done_count = 0
        self.df_last = None
        self.df_now = None
        return self.state  # 返回初始状态

    def init_state_to_zero(self):
        # 随机状态
        # random_vector = [random.randint(self.lb[i], self.ub[i]) for i in range(len(self.lb))]
        # random_state = np.array(random_vector)
        # return random_state
        self.daXY_history = np.array(self.df_class.orign_df.copy().dropna())
        self.dataX_history = self.daXY_history[:,:-1]
        self.dataY_history = self.daXY_history[:,-1]
        return np.array([0] * self.feature_num * 3)
    def init_state_random(self):
        # 随机状态
        self.dataXY_old = None
        random_vector = [random.randint(self.lb[i], self.ub[i]) for i in range(len(self.lb))]
        random_state = np.array(random_vector)
        return random_state



    def is_valid_action(self, action, state):
        """
        检查给定动作是否对当前状态有效。
        """
        # 实现动作有效性的检查逻辑
        # 例如: 如果是 c 参数的动作，确保不会增加已经是 1 的 c 或减少已经是 0 的 c

        # 示例逻辑（需要根据你的环境具体实现进行修改）
        param_type = action // (2 * self.feature_num)  # 确定是 t, tao 还是 c
        is_increment = action % 2 == 1  # 判断是否为增加参数的动作
        feature_index = (action // 2) % self.feature_num  # 确定是哪一个特征的参数
        if param_type < 2:  # 对于 t 或 tao 的操作
            # 首先检查 c 是否允许对 t 或 tao 进行操作
            c_value = state[2 * self.feature_num + feature_index]
            if c_value == 0:
                return False  # 如果 c 为 0，则对 t 或 tao 的任何操作都是无效的
            else :
                if is_increment :
                    if state[param_type* self.feature_num + feature_index] +1 > self.ub[param_type* self.feature_num + feature_index] :
                        return False
                else:
                    if state[param_type* self.feature_num + feature_index] -1 < self.lb[param_type* self.feature_num + feature_index] :
                        return False

        elif param_type == 2:  # 假设 t, tao, c 分别对应 0, 1, 2
            current_c_value = state[2 * self.feature_num + feature_index]
            if (current_c_value == 1 and is_increment) or (current_c_value == 0 and not is_increment):
                return False  # 动作无效

        return True  # 动作有效

    def step(self, action):
        """
        执行一个动作，并返回新状态、奖励、完成标志和调试信息
        """
        # 得到特征的下标
        feature_index = (action // 2) % self.feature_num
        is_valid = self.is_valid_action(action,self.state)

        if is_valid:
            # 判断动作属于t、tao还是c的动作
            if action < 2 * self.feature_num:  # t 的动作
                if action % 2 == 0:# 减少 t
                    self.state[feature_index] = max(0, int(self.state[feature_index] - 1))
                else:#增加t
                    self.state[feature_index] = self.state[feature_index] + 1  # 增加 t
            elif action < 4 * self.feature_num:  # tao 的动作
                feature_index -= self.feature_num
                if action % 2 == 0:# 减少 tao
                    self.state[self.feature_num + feature_index] = max(0, int(self.state[self.feature_num + feature_index] - 1))
                else:# 增加tao
                    self.state[self.feature_num + feature_index] = self.state[self.feature_num + feature_index] + 1  # 增加 tao
            else:  # c 的动作
                self.state[2 * self.feature_num + feature_index] = action % 2  # 设置 c
            reward = self.get_reward(self.state,action)
        else :
            self.done_count += 1
            reward = self.invalid_action_reward

        # 判断是否连续n次reward提升都很小，如果是就停止迭代
        logging.debug(f"is_valid: {is_valid},count: {self.done_count },reward:{reward}")
        done = True if self.done_count > self.max_stop_step else False
        # 调试信息可以包括额外的数据，这里我们不包括任何额外信息
        info = {}

        # 返回新的状态，奖励，是否完成，以及调试信息
        return self.state, reward, done, info

