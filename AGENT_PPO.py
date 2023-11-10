import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from data_process import varSelection, dataShift, dataFilter
from model_evaluation import R2, calculate_best_R2

class FSEnv():
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


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


