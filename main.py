# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
from Env import Environment
from Env import Agent
from Env import findbestfeatures
from data_process import importData
import numpy as np

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    dataframe = importData("/Users/somac/Library/Mobile Documents/com~apple~CloudDocs/Code/SoftSensor_python/data/data05.csv",
                   "/Users/somac/Library/Mobile Documents/com~apple~CloudDocs/Code/SoftSensor_python/data/T35111A.csv")
    data = np.array(dataframe)[:,:-1]
    fsenv = Environment(data)
    fsagent = Agent(data.shape[1]-1)
    AOR = findbestfeatures(fsagent,fsenv)

    print(AOR)
    # FS = Environment('./')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
