# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
from Env import Environment
from Env import Agent
from Env import findbestfeatures
from data_process import importData
import numpy as np
import random

random.seed(2023)
np.random.seed(2023)
xPath = "./data/data05.csv"
yPath = "./data/T35111A.csv"
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    dataframe = importData(xPath,yPath)
    data = np.array(dataframe)[:,:-1]
    FS_env = Environment(data)
    FS_agent = Agent(data.shape[1]-1,2,reward_K=100)
    AOR = findbestfeatures(FS_agent,FS_env,1000000)

    print(np.array2string(AOR, precision=5, suppress_small=True))
    # FS = Environment('./')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
