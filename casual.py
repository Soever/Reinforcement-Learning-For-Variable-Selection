from data_process import importData,XPATH,YPATH,DataClass
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.feature_selection import mutual_info_regression
import lingam
import networkx as nx
from lingam.utils import make_prior_knowledge, make_dot
if __name__ == "__main__":
    DFcls = DataClass()
    MAX_DELAY = 480

    temp_data = DFcls.orign_df.dropna()
    model = lingam.DirectLiNGAM()
    model.fit(temp_data)
    a = model.adjacency_matrix_
    make_dot(model.adjacency_matrix_)



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
