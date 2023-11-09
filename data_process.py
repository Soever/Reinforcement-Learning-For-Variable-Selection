import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import StandardScaler
import warnings
X1PATH = 'data/data05.csv'
Y1PATH = 'data/T35111A.csv'
X2PATH = 'data/2017/data2017412-20170609.csv'
Y2PATH = 'data/2017/T35111A20170412-20170609.csv'
XPATH2016merge = "data/2016/mergin/2016_x_data.csv"
YPATH2016merge = "data/2016/mergin/2016_y_data.csv"
XPATH2016CLEAN = "data/2016/clean/data20161018-1112.csv"
YPATH2016CLEAN = "data/2016/clean/T35111A.csv"
class DataClass():
    def __init__(self, xDataPath=X1PATH, yDataPath=Y1PATH,drop_last_col=None,labindex=None):
        """
        :param xDataPath:
        :param yDataPath:
        :param labindex:y变量中的第几列是当前建模变量
        :param drop_last_col: 是否舍弃最后一列
        """
        self.Y_num = None
        self.feature_num = None

        self.xDataPath = xDataPath
        self.yDataPath = yDataPath
        self.orign_df  = self.importData(drop_last_col)


        if labindex is not None:
            self.lab_index =self.Y_num+labindex-1
        else :
            self.lab_index = 1
        # 生成numpy数据

        self.orign_X   = np.array(self.orign_df.iloc[:,:-self.lab_index])
        self.orign_Y = np.array(self.orign_df.iloc[:, -self.lab_index].dropna()).reshape(-1,1)
        #计算样本数
        self.lab_samples_num = self.orign_df.iloc[:, -1].notna().sum()
        # 初始化标准化工具
        self.scalerX = StandardScaler().fit(self.orign_X )
        self.scalerY = StandardScaler().fit(self.orign_Y)
        # 生成标准化数据
        self.orign_X_std = self.scalerX.transform(self.orign_X)
        self.orign_Y_std = self.scalerY.transform(self.orign_Y)

    def importData(self,drop):
        """

        :param xDataPath:
        :param yDataPath:
        :param mode: mode=1 : 得到有标签样本行的X，Y
                     mode=2 ：
        :return:
        """
        # 读取数据集X，Y
        xPath = os.path.join(os.path.dirname(__file__),self.xDataPath)
        yPath = os.path.join(os.path.dirname(__file__),self.yDataPath)

        dfx = pd.read_csv(xPath, skipinitialspace=True, low_memory=False)
        dfy = pd.read_csv(yPath, skipinitialspace=True, low_memory=False)

        # 读取X采样时间
        timex = dfx.loc[0, "Time"]

        # 删除X，Y前两行
        dfx.drop(dfx.head(2).index, inplace=True)
        dfx = dfx.reset_index(drop=True)
        dfy.drop(dfx.head(2).index, inplace=True)
        dfy = dfy.reset_index(drop=True)

        # 将Time转为时间戳
        dfx['Time'] = pd.to_datetime(dfx['Time'])
        dfy['Time'] = pd.to_datetime(dfy['Time'])

        # 将Time设置为索引
        dfx = dfx.set_index('Time')
        dfy = dfy.set_index('Time')
        self.feature_num  = dfx.shape[1]
        self.Y_num= dfy.shape[1]
        # 将y采样变成30s，其余时间变成NAN
        dfy = dfy.resample('30S').asfreq()
        # 按照时间戳合并X，Y
        df = pd.merge(left=dfx, right=dfy, how='outer', on='Time')
        df = df.dropna(subset=df.columns[:-2])
        if drop is not None:
            df= df.iloc[:, :-1]
            self.Y_num -= 1
        # 重置索引为默认索引
        # df = df.reset_index()

        return df.astype(float)

    def get_Shift_data(self,shiftList):
        tmpDF = self.orign_df.copy()
        # 对每行分别做延迟
        for i in range(len(shiftList)):
            tmpDF.iloc[:, i] = tmpDF.iloc[:, i].shift(int(shiftList[i]))
        tmpDF.dropna(inplace=True)
        return tmpDF

def excel2np(path):
    df = pd.read_excel(path, header=[0, 1])
    # 转换为 Numpy 数组
    array = df.values[:,1:]
    X = array[:,:-3]
    Y = array[:,-3:]
    return X,Y

def normalize(X, Y):
    """
    :param X: 原始数据X
    :param Y: 原始数据Y
    :return: 标准化数据x，标准化数据y，均值列表，标准差列表
    """
    # 初始化均值、标准差列表、标准X、标准Y
    mean = []
    STD = []
    stdX = np.empty_like(X)
    stdY = np.empty_like(Y)
    # 每列标准化
    for i in range(X.shape[1]):
        # 求均值mean
        m = np.mean(X[:, i])
        # 求方差var
        v = np.var(X[:, i])
        # 求标准差std
        v = v ** 0.5
        mean.append(m)
        STD.append(v)
        # 标准化 X =  (x-mean)/std
        stdX[:, i] = (X[:, i] - m) / v
    # 求均值mean
    m = np.mean(Y)
    # 求标准差std
    v = np.var(Y)
    v = v ** 0.5
    mean.append(m)
    STD.append(v)
    # 标准化Y
    stdY = (Y - m) / v

    return stdX, stdY, mean, STD

def importData(xDataPath=X1PATH, yDataPath=Y1PATH, mode = 1):
    """

    :param xDataPath:
    :param yDataPath:
    :param mode: mode=1 : 得到有标签样本行的X，Y
                 mode=2 ：
    :return:
    """
    # 读取数据集X，Y
    xPath = os.path.join(os.path.dirname(__file__), xDataPath)
    yPath = os.path.join(os.path.dirname(__file__), yDataPath)
    dfx = pd.read_csv(xPath,skipinitialspace=True, low_memory=False)
    dfy = pd.read_csv(yPath, skipinitialspace=True,low_memory=False)

    # 读取X采样时间
    timex = dfx.loc[0, "Time"]

    # 删除X，Y前两行
    dfx.drop(dfx.head(2).index, inplace=True)
    dfx = dfx.reset_index(drop=True)
    dfy.drop(dfx.head(2).index, inplace=True)
    dfy = dfy.reset_index(drop=True)

    # 将Time转为时间戳
    dfx['Time'] = pd.to_datetime(dfx['Time'])
    dfy['Time'] = pd.to_datetime(dfy['Time'])
    # 将Time设置为索引
    dfx = dfx.set_index('Time')
    dfy = dfy.set_index('Time')
    # 将y采样变成30s，其余时间变成NAN
    dfy = dfy.resample('30S').asfreq()
    # 按照时间戳合并X，Y
    df = pd.merge(left=dfx, right=dfy, how='outer', on='Time')

        # 除掉包含任何包含Nan的行
    df = df.dropna(axis=0, how='any')
    df = df.dropna(subset=df.columns[:-2]).iloc[:, :-1]
    # 重置索引为默认索引
    # df = df.reset_index()
    return df


def save(ind:list):
    now = datetime.datetime.now()
    time_suffix = now.strftime('%Y%m%d%H%M')
    filename = f'./result/{time_suffix}.txt'
    with open(filename, 'w') as f:
        # 保存numpy数组
        for i in ind:
            if isinstance(i, np.ndarray):
                np.savetxt(f, i, fmt='%s')
                f.write("\n")

            # 保存整数
            if isinstance(ind, int):
                f.write(f"{i}\n")

            # 保存numpy浮点数
            if isinstance(ind, (np.float64, float)):
                f.write(f"{i}\n")
    return filename

def dataShift(df, shiftList):
    """
    对数据做延迟操作，往后移为正数
    df: 变量选择后的dataframe
    shiftList: 对应变量的延迟参数
    """
    # 复制原始表格
    tmpDF = df.copy()
    # 对每行分别做延迟
    for i in range(len(shiftList)):
        tmpDF.iloc[:, i] = tmpDF.iloc[:, i].shift(int(shiftList[i]))
    # 删除延迟后表格的空缺部分
    # tmpDF.dropna(inplace=True)
    return tmpDF

def dataFilter(df, filter_values):
    """
    对数据做滤波操作
    df: 变量选择、延迟后的dataframe
    filter_values: 滤波值
    """
    # 将表格设置为行索引，原始为时间索引
    tmpdf = df.reset_index(drop=True)
    # 得到不包含任何NAN行的索引，即有标签的样本的索引
    idxlist = tmpdf.dropna(axis=0, how='any').index.to_list()
    # 从有标签的第一行开始，转化为numpy类型，得到辅助变量数据
    x_origin = np.array(tmpdf.iloc[idxlist[0]:idxlist[-1]+1,:-1])
    # 得到第一个有标签行的索引，将索引更新以便与nupmy数据对应，因为numpy数据删了表格第一行以前的数据
    a = idxlist[0]
    idxlistx = [x - a for x in idxlist]
    # 根据索引得到表格里的y数据
    y = np.array(tmpdf.iloc[idxlist,-1])
    # 计算滤波系数
    # 时间常数越大，系统惯性越强，越取决于上一时刻的输出
    delta = 1e-10
    alpha = np.zeros_like(filter_values, dtype=float)
    non_zero_indices = filter_values != 0
    alpha[non_zero_indices] = np.exp(-1 / filter_values[non_zero_indices])
    # 使用该警告处理函数

    # 计算x滤波值
    filtered_x = np.zeros_like(x_origin)
    filtered_x[0, :] = x_origin[0, :]
    for i in range(1, x_origin.shape[0]):
        filtered_x[i, :] = alpha * filtered_x[i - 1, :] + (1 - alpha) * x_origin[i, :]
    # 得到有标签样本
    x = filtered_x[idxlistx]
    return x, y

def varSelection(df, selector):
    tmpDF = df.copy(deep=True)
    col = []
    # 如果值为0表示未被选中，则从表格中删除该列
    for i in range(len(selector)):
        if selector[i] == 0:
            col.append(i)
    tmpDF.drop(df.columns[col], axis=1, inplace=True)
    return tmpDF