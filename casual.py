from data_process import importData,XPATH,YPATH,DataClass
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.feature_selection import mutual_info_regression
import lingam
import networkx as nx
if __name__ == "__main__":
    DFcls = DataClass()
    MAX_DELAY = 480
    cor = np.zeros((MAX_DELAY,DFcls.features_num))
    temp_data = DFcls.orign_df.dropna()
    model = lingam.DirectLiNGAM()
    model.fit(temp_data)
    causal_matrix = model.adjacency_matrix_

    G = nx.DiGraph(causal_matrix)
    source_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]

    # 输出源头节点
    print("Source nodes:", source_nodes)
    # 为图的节点添加标签
    labels = {i: col for i, col in enumerate(temp_data.columns)}

    # 使用Matplotlib绘制图形
    pos = nx.spring_layout(G)
    nx.draw(G, pos, labels=labels, with_labels=True, node_color='skyblue', node_size=3000, edge_color='gray',
            linewidths=2, font_size=15, arrowsize=20, connectionstyle='arc3,rad=0.1')
    plt.title('Causal Graph')
    plt.show()
    # for i in range(MAX_DELAY):
    #     delaylist = [i] * DFcls.features_num
    #     temp_data =np.array(DFcls.get_Shift_data(delaylist))
    #     stdx = DFcls.scalerX.transform(temp_data[:,:-1])
    #     stdy = DFcls.scalerY.transform(temp_data[:,-1].reshape(-1,1))
    #
    #     for feature_index in range(DFcls.features_num):
    #         x_data = stdx
    #         y_data = stdy.reshape(-1)
    #         #计算person相关系数
    #         # cor[i,feature_index] = np.corrcoef(x_data[:,feature_index ], y_data)[0, 1]
    #         # 计算相关互信息
    #
    #         cor[i, feature_index]= mutual_info_regression(x_data[:,feature_index ].reshape(-1,1),y_data)[0]
    # cor
    # X= cor
    # plt.figure(figsize=(15, 10))
    # colors = plt.cm.tab10(np.linspace(0, 1, DFcls.features_num))
    # # 绘制8张图
    # for i in range( DFcls.features_num):  # 选择8和特征数m中的较小值作为绘图的上限
    #     plt.subplot(3, 3, i + 1)  # 2行4列的子图布局
    #     plt.plot(X[:, i],  color=colors[i],label=f'Feature {i + 1}')  # 绘制每个特征的图形
    #     plt.annotate(f'Feature {i + 1}',
    #                  (0, X[0, i]),  # 文本的位置
    #                  textcoords="offset points",  # 文本位置的参照坐标系
    #                  xytext=(10, 10),  # 文本的偏移量
    #                  ha='center',  # 横向对齐方式
    #                  color=colors[i])  # 文本颜色
    #     # plt.ylim([0.1, 0.59])
    #     plt.legend()  # 添加图例
    #     plt.title(f'Mutual Information of Feature {i + 1}')  # 设置标题
    #     # plt.title(f'Mutual Information of All Features')
    #     plt.xlabel('time delay parameter')  # 设置x轴标签
    #     plt.ylabel('Mutual Information')  # 设置y轴标签
    #
    # # 调整子图之间的间距并显示图形
    # plt.tight_layout()
    # plt.show()