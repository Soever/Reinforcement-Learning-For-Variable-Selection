import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义DQN网络模型
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 128)          # 第二个全连接层
        self.fc3 = nn.Linear(128, n_actions)    # 第三个全连接层，输出动作的Q值

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层激活函数为ReLU
        x = F.relu(self.fc2(x))  # 第二层激活函数为ReLU
        return self.fc3(x)       # 输出层不加激活函数，直接输出Q值

# 测试网络，假设有一个4维的观测空间和两个可能的动作
test_net = DQN(4, 2)
test_observation = torch.FloatTensor([[1, 2, 3, 4]])
print("Output Q-values for the test observation:", test_net(test_observation))
