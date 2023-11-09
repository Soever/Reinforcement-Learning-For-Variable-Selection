from collections import namedtuple, deque
import random

# 定义一个经验元组
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

# 测试记忆回放
test_memory = ReplayBuffer(1000)
test_experience = Experience([1,2,3,4], 1, 1.0, [1,2,3,5], False)
test_memory.push(*test_experience)

print("Number of experiences in memory:", len(test_memory))
print("Sampled experience:", test_memory.sample(1))
