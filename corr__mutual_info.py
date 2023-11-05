from data_process import importData,X1PATH,Y1PATH,X2PATH,Y2PATH,DataClass
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.feature_selection import mutual_info_regression


MAX_DELAY = 240

if __name__ == "__main__":
    DFcls = DataClass(xDataPath=X2PATH,
                      yDataPath=Y2PATH,
                      labindex= 1,
                      drop_last_col=True,
                      )

    cor = np.zeros((MAX_DELAY,DFcls.features_num))
    for i in range(MAX_DELAY):
        delaylist = [i] * DFcls.features_num
        temp_data =np.array(DFcls.get_Shift_data(delaylist))
        stdx = DFcls.scalerX.transform(temp_data[:,:-1])
        stdy = DFcls.scalerY.transform(temp_data[:,-1].reshape(-1,1))

        for feature_index in range(DFcls.features_num):
            x_data = stdx
            y_data = stdy.reshape(-1)
            #计算person相关系数
            # cor[i,feature_index] = np.corrcoef(x_data[:,feature_index ], y_data)[0, 1]
            # 计算相关互信息

            cor[i, feature_index]= mutual_info_regression(x_data[:,feature_index ].reshape(-1,1),y_data)[0]

    plt.figure(figsize=(30, 15))
    colors = plt.cm.tab10(np.linspace(0, 1, DFcls.features_num))
    for i in range(DFcls.features_num):  # 选择8和特征数m中的较小值作为绘图的上限
        plt.plot(cor[:, i], color=colors[i], label=f'Feature {i + 1}')  # 绘制每个特征的图形
        plt.annotate(f'Feature {i + 1}',
                     (0, cor[0, i]),  # 文本的位置
                     textcoords="offset points",  # 文本位置的参照坐标系
                     xytext=(10, 10),  # 文本的偏移量
                     ha='center',  # 横向对齐方式
                     color=colors[i])  # 文本颜色
        # plt.ylim([0.1, 0.59])
        plt.legend()  # 添加图例
        plt.title(f'MI of All Features')  # 设置标题
        # plt.title(f'Mutual Information of All Features')
        plt.xlabel('time delay parameter')  # 设置x轴标签
        plt.ylabel('MI')  # 设置y轴标签

    # 调整子图之间的间距并显示图形
    plt.show()

    #分别绘制每张图
    plt.figure(figsize=(30, 15))
    colors = plt.cm.tab10(np.linspace(0, 1, DFcls.features_num))
    for i in range( DFcls.features_num):
        plt.subplot(6, 6, i + 1)
        plt.plot(cor[:, i], color=colors[i])  # 绘制每个特征的图形

        # plt.ylim([0.1, 0.59])

        plt.title(f'PCCs of Feature {i + 1}')  # 设置标题
        # plt.title(f'Mutual Information of All Features')
        plt.xlabel('time delay parameter')  # 设置x轴标签
        plt.ylabel('PCCs')  # 设置y轴标签

    # 调整子图之间的间距并显示图形
    plt.tight_layout()
    plt.show()