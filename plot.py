from data_process import X1PATH,Y1PATH,DataClass,X2PATH,Y2PATH
import numpy as np
import matplotlib.pyplot as plt
from rl_utils import moving_average
import os
def plot_PPO(file_path):

    if isinstance(file_path, np.ndarray):
        return_list = file_path
    elif isinstance(file_path, str) and os.path.isfile(file_path) and file_path.endswith('.txt'):
    # 从文本文件加载数据
        return_list = np.loadtxt(file_path, dtype=float)
    else:
        print("格式不正确")
        return
    # 计算移动平均
    mv_return = moving_average(return_list, 19)  # 使用窗口大小为9

    # 准备绘图
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(12, 6))

    # 绘制实际回报，使用非常浅的颜色
    plt.plot(episodes_list, return_list, color='blue', alpha=0.2, label='Actual Returns')

    # 绘制移动平均，使用较深的颜色
    plt.plot(episodes_list, mv_return, color='blue', alpha=0.9, label='Moving Average')

    # 设置坐标轴标签和图表标题
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on Variable Selection')

    # 显示图例
    plt.legend()

    # 显示图表
    plt.show()


def round_to_three_significant_figures(num):
    """
    Rounds a number to have three significant figures.
    """
    # Determine the factor to shift the decimal to the rightmost significant digit
    factor = 10 ** (2 - int(np.floor(np.log10(abs(num)))))

    # Round the number and then shift back
    return round(num * factor) / factor
if __name__ == "__main__":
    # DFcls = DataClass(xDataPath="/Users/somac/同步空间/软测量/matlab source/simulink model/1009/generate_data/60s_100to2260_003/dataX_no_noise.csv",
    #                   yDataPath="/Users/somac/同步空间/软测量/matlab source/simulink model/1009/generate_data/60s_100to2260_003/dataY_no_noise_4H.csv",
    #                   labindex= 1,
    #                   drop_last_col=True,
    #                   )
    # df = DFcls.orign_df
    # df= df.reset_index(drop=True)
    # i = 0
    # plt.figure(figsize=(20, 15))
    # for column in df.columns:
    #     if i < 22:
    #         plt.subplot(13, 2, i + 1)# 设置图形大小
    #
    #         df[column].plot(kind='line', title=f'{column}')  # 绘制折线图，您可以更改为其他图形类型，如'bar', 'hist'等
    #         ymin, ymax = plt.ylim()
    #
    #         # 设置y轴的刻度标签并调整大小
    #         plt.yticks([round(ymin,1), round(ymax,1)],fontsize = 7)
    #         plt.xticks([])
    #     i+=1# 显示图表
    # plt.show()
    # 绘制PPO训练结果
    return_list = np.loadtxt('./result/PPO_1116reward.txt', dtype=float)
    plot_PPO(return_list)

