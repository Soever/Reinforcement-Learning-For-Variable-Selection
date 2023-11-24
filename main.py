# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
from Env import Environment,Agent,find_best_features
from data_process import importData,X1PATH,Y1PATH,X2PATH,Y2PATH,DataClass,XPATH2016merge,YPATH2016merge,XPATH2016CLEAN,YPATH2016CLEAN
import numpy as np
import rl_utils
import torch
import random
import matplotlib.pyplot as plt
from AGENT_PPO import PPO
from ENV2 import FSEnv
from plot import plot_PPO
import logging

from AGENT_DQN import DQNAgent,DQNEnv,train_dqn
def set_seed(seed_value=2023):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True



def q_table_learn(df_class):
    dfnew = df_class.orign_df.dropna(axis=0, how='any')
    # dataframe = importData(X1PATH, Y1PATH)
    data = np.array(dfnew)
    FS_env = Environment(data,
                         reward_K=1)  # 奖励放大系数 ，由于增加一个特征带来的R2普遍偏低时可设置，目前不需要

    FS_agent = Agent(
        m=data.shape[1] - 1,  # 特征个数
        policy=1,  # 搜寻策略 0：随机  1：贪婪  2：e-贪婪
        epsilon=0.25,  # policy = 2 时 参数e的值
        pre_evaluate=False,  # 是否预评估
        AOR=None  # 初始特征分数
    )

    AOR = find_best_features(FS_agent, FS_env, 10000)

    print(np.array2string(AOR, precision=5, suppress_small=True))
    # FS = Environment('./')

def DQN_learn(df_class):
    # 设置参数
    state_size = df_class.feature_num *3 # 每个变量有3个参数
    action_size = df_class.feature_num *6 # 每个参数有两个动作，增或减
    replay_memory_size = 1000
    batch_size = 20
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.98
    lr = 0.01
    num_episodes = 1000


    # 创建环境和智能体
    env = DQNEnv(df_class,state_size, action_size)
    agent = DQNAgent(state_size, action_size, replay_memory_size, batch_size, gamma, epsilon, epsilon_min,
                     epsilon_decay, lr)


    # 训练DQN智能体
    train_dqn(env, agent, num_episodes)

def PPO_learn(df_class):
    state_size = df_class.feature_num * 3  # 每个变量有3个参数
    action_size = df_class.feature_num * 6  # 每个参数有两个动作，增或减
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = FSEnv(df_class=df_class, state_size=state_size, action_size=action_size,
                invalid_action_reward = -0.1 , # 违反约束时的奖励
                min_score =  0, #视为有提升的最小阈值
                max_stop_step=state_size,# 最大停滞步数 智能体n步都不提升时停止
                )
    # env.seed(0)
    torch.manual_seed(2023)
    state_dim = state_size
    action_dim = action_size
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)


    plot_PPO(np.array(return_list))

    with open('./result/PPO_1124.txt', 'w') as file:
        for item in return_list:
            file.write(f"{item}\n")


if __name__ == '__main__':
    set_seed(2023)
    # df_class = DataClass(XPATH2016merge,YPATH2016merge,
    #                drop_last_col=None,
    #                labindex=None)
    logging.basicConfig(filename='result/reward-1_debuglog.log', level=logging.DEBUG)
    df_class = DataClass(XPATH2016CLEAN, YPATH2016CLEAN,
                         drop_last_col=True,
                         labindex=None)
    #q_table_learn(df_class)
    #DQN_learn(df_class)

    PPO_learn(df_class)



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
