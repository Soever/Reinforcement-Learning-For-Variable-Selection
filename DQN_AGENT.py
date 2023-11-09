import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
# from DQN.Replay import ReplayBuffer,Experience
# from DQN.DQN import DQN

from data_process import varSelection, dataShift, dataFilter
from data_process import normalize
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from model_evaluation import R2, calculate_best_R2
from sklearn.metrics import r2_score
import sys
import random
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 128)  # 第二个全连接层
        self.fc3 = nn.Linear(128, n_actions)  # 第三个全连接层，输出动作的Q值

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层激活函数为ReLU
        x = F.relu(self.fc2(x))  # 第二层激活函数为ReLU
        return self.fc3(x)  # 输出层不加激活函数，直接输出Q值


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)  # 使用deque来保存经验，当达到最大容量时自动丢弃旧的经验

    def push(self, *args):
        """保存一个transition"""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """随机抽取一些经验作为一个batch进行学习"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """返回当前记忆中的经验数量"""
        return len(self.memory)


class DQNAgent:
    """DQN智能体类"""

    def __init__(self, state_size, action_size, replay_memory_size, batch_size, gamma, epsilon, epsilon_min,
                 epsilon_decay, lr):
        self.state_size = state_size  # 状态空间大小
        self.action_size = action_size  # 动作空间大小
        self.feature_num = state_size / 3
        self.memory = ReplayBuffer(replay_memory_size)  # 创建记忆回放缓存
        self.batch_size = batch_size  # 批量大小
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # ε-贪婪策略中的ε值
        self.epsilon_min = epsilon_min  # ε的最小值
        self.epsilon_decay = epsilon_decay  # ε的衰减率
        self.model = DQN(state_size, action_size)  # 创建DQN模型
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # 使用Adam优化器

    def select_action(self, state):
        """根据当前状态选择动作"""
        if random.random() > self.epsilon:  # 判断是否要执行探索
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state)
                return np.argmax(q_values.cpu().data.numpy())
        else:
            return random.randrange(self.action_size)

    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """从记忆中学习"""
        if len(self.memory) < self.batch_size:  # 确保有足够的经验来学习
            return

        experiences = self.memory.sample(self.batch_size)  # 随机获取经验批次
        batch = Experience(*zip(*experiences))

        states = torch.FloatTensor(np.array(batch.state))  # # 将列表转换为单个 NumPy 数组,然后将 NumPy 数组转换为 PyTorch 张量

        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(batch.done)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




class DQNEnv():
    def __init__(self, df_class, state_size, action_size, update_method=1, reward_K=1):
        self.df_class = df_class

        self.feature_num = self.df_class.feature_num
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.array([0] * self.feature_num * 3)  # 当前状态
        self.action = np.array([0] * self.feature_num * 3 * 2)  # 当前状态可以采取的动作，1为可以采取，0不可采取

        self.acc_new = 0
        self.acc_old = 0
        self.best_R2 = 0

        self.best_state = None

        self.lb = [0] * self.feature_num + [0] * self.feature_num + [0] * self.feature_num  # 决策变量下界
        self.ub = [60] * self.feature_num + [60] * self.feature_num + [1] * self.feature_num  # 决策变量上界
        self.done_count = 0

    def init(self):
        self.acc_old = 0
        self.acc_new = 0

    def get_reward(self, state):
        """
        :param state:当前状态
        :param action_t: 当前状态采取的动作
        :return: 采取当前动作后准确度得到的提升
        """
        # 计算该特征集合下的准确度
        self.acc_new = self.calculate_accuracy(state)

        # 计算和上个状态的差值
        acc_diff = self.acc_new - self.acc_old
        if acc_diff < 0.005:
            self.done_count += 1
        else:
            self.done_count = 0
        # 保存该状态下的准确度
        self.acc_old = self.acc_new
        return acc_diff

    def calculate_accuracy(self, state):
        delayvalue = state[:self.feature_num]
        filtervalue = state[self.feature_num:2 * self.feature_num]
        varSelect = state[2 * self.feature_num:].astype(bool)
        p = varSelect.sum()
        originDF = self.df_class.orign_df
        bestR2 = 0
        if p > 0:
            # 选择变量
            df = varSelection(originDF, varSelect)
            # 处理滞后
            df = dataShift(df, delayvalue[varSelect])
            # 一阶滤波
            X, y = dataFilter(df, filtervalue[varSelect])
            bestR2, latent_var_num = calculate_best_R2(X, y)

        if self.best_R2 < bestR2:
            self.best_R2 = bestR2
            self.best_state = state
            # print("最好的R2:" + str(self.best_R2))
            # print("当前最优的状态为：" + str(self.best_state))

        return bestR2

    def reset(self):
        """
        重置环境到初始状态
        """
        self.state = self.init_state()  # 随机生成state
        self.acc_old = 0
        self.best_R2 = 0
        self.best_state = None
        self.done_count = 0
        return self.state  # 返回初始状态

    def init_state(self):
        # 随机状态
        # random_vector = [random.randint(self.lb[i], self.ub[i]) for i in range(len(self.lb))]
        # random_state = np.array(random_vector)
        # return random_state
        return np.array([0] * self.feature_num * 3)

    def step(self, action):
        """
        执行一个动作，并返回新状态、奖励、完成标志和调试信息
        """

        feature_index = (action // 2) % self.feature_num
        if action < 2 * self.feature_num:  # t 的动作
            if action % 2 == 0:
                self.state[feature_index] = max(0, int(self.state[feature_index] - 1))  # 减少 t
            else:
                self.state[feature_index] = self.state[feature_index] + 1  # 增加 t
        elif action < 4 * self.feature_num:  # tao 的动作
            feature_index -= self.feature_num
            if action % 2 == 0:
                self.state[self.feature_num + feature_index] = max(0, int(
                    self.state[self.feature_num + feature_index] - 1))  # 减少 tao
            else:
                self.state[self.feature_num + feature_index] = self.state[self.feature_num + feature_index] + 1  # 增加 tao
        else:  # c 的动作
            self.state[2 * self.feature_num + feature_index] = action % 2  # 设置 c

        # 计算这次动作后的奖励
        reward = self.get_reward(self.state)

        # 判断是否连续n次reward提升都很小，如果是就停止迭代
        done = True if self.done_count > 20 else False

        # 调试信息可以包括额外的数据，这里我们不包括任何额外信息
        info = {}

        # 返回新的状态，奖励，是否完成，以及调试信息
        return self.state, reward, done, info

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
        elif param_type == 2:  # 假设 t, tao, c 分别对应 0, 1, 2
            current_c_value = state[2 * self.feature_num + feature_index]
            if (current_c_value == 1 and is_increment) or (current_c_value == 0 and not is_increment):
                return False  # 动作无效

        return True  # 动作有效


def train_dqn(env, agent, num_episodes):
    """DQN训练循环"""
    for episode in range(num_episodes):
        state = env.reset()  # 重置环境，获取初始状态
        total_reward = 0
        done = False
        best_reward = 0
        count = 0
        while not done:
            action = agent.select_action(state)  # 根据当前状态选择动作
            count  += 1
            if not env.is_valid_action(action, state):
                # 如果动作无效，给出负奖励，并可能结束回合或重新选择动作
                reward = -1  # 设定一个负奖励
                agent.store_experience(state, action, reward, state, False)  # 存储经验，这里没有结束回合，所以done为False
                if count > 20:
                    agent.learn()  # 学习
                    count = 0
                total_reward += reward  # 累计奖励

                continue  # 跳过执行无效动作，重新选择动作
            next_state, reward, done, _ = env.step(action)  # 执行动作，获取下一状态和奖励

            agent.store_experience(state, action, reward, next_state, done)  # 存储经验
            if count > 20:
                agent.learn()  # 学习
                count = 0
            state = next_state  # 更新状态
            total_reward += reward  # 累计奖励
            if total_reward > best_reward:
                best_reward = total_reward
                print(best_reward)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay  # 更新epsilon进行衰减，实现探索与利用的平衡

        print(
            f'Episode: {episode}, Total reward: {total_reward},best reward: {best_reward} Best R2: {env.best_R2}, Epsilon: {agent.epsilon:.2f}')  # 打印信息
