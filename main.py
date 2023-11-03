# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
from Env import Environment
from Env import Agent
from Env import find_best_features
from data_process import importData,XPATH,YPATH,X2PATH,Y2PATH,DataClass,XPATH2016,YPATH2016
import numpy as np

import torch
import random

def set_seed(seed_value=2023):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    set_seed(2023)


    dataframe = DataClass(XPATH2016, YPATH2016).orign_df
    data = np.array(dataframe.dropna())
    FS_env = Environment(data,
                         reward_K = 1) # 奖励放大系数 ，由于增加一个特征带来的R2普遍偏低时可设置，目前不需要

    FS_agent = Agent(m=data.shape[1]-1,  # 特征个数
             policy=2,           # 搜寻策略 0：随机  1：贪婪  2：e-贪婪
             epsilon=0.6,       # policy = 2 时 参数e的值
             pre_evaluate=False, # 是否预评估
             AOR=None            # 初始特征分数
             )
    AOR = find_best_features(FS_agent,FS_env,5000)

    print(np.array2string(AOR, precision=5, suppress_small=True))
    # FS = Environment('./')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
