import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
X1PATH = './data/data05.csv'
Y1PATH = './data/T35111A.csv'
X2PATH = './data/2017/data2017412-20170609.csv'
Y2PATH = './data/2017/T35111A20170412-20170609.csv'

class DataClass():
    def __init__(self, xDataPath=X1PATH, yDataPath=Y1PATH,labindex=None,drop_last_col=None):
        """
        :param xDataPath:
        :param yDataPath:
        orign_X(n,m)
        orign_Y(n,1)
        """
        self.Y_num = None
        self.features_num = None

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
        #计算特征数、样本数

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
        dfx = pd.read_csv(self.xDataPath, skipinitialspace=True, low_memory=False)
        dfy = pd.read_csv(self.yDataPath, skipinitialspace=True, low_memory=False)

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
        self.features_num  = dfx.shape[1]
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
    dfx = pd.read_csv(xDataPath,skipinitialspace=True, low_memory=False)
    dfy = pd.read_csv(yDataPath, skipinitialspace=True,low_memory=False)

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