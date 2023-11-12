import pandas as pd


if __name__ == "__main__":
    dfx1 = pd.read_csv("data/2016/orgin/data20161018-1112.csv",skipinitialspace=True,parse_dates=[0], low_memory=False)
    dfx2 = pd.read_csv("data/2016/orgin/data20161112-1125.csv", skipinitialspace=True,parse_dates=[0],low_memory=False)
    dfx3 = pd.read_csv("data/2016/orgin/data20161125-1223--.csv", skipinitialspace=True,parse_dates=[0],low_memory=False)
    #
    dfy1 = pd.read_csv("data/2016/orgin/T35111A.csv", skipinitialspace=True,parse_dates=[0],low_memory=False)
    dfy2 = pd.read_csv("data/2016/orgin/T35111A(201612).csv", skipinitialspace=True,parse_dates=[0],low_memory=False)

    dfx = pd.concat([dfx1, dfx2, dfx3])
    dfxx = dfx.iloc[:,0:-9]
    columns_to_convert = dfxx.columns[1:]  # 假设第一列是日期列，我们从第二列开始转换


    for column in columns_to_convert:
        dfxx[column] = dfxx[column].astype(float)

    dfx = dfxx

    dfy = pd.concat([dfy1, dfy2])
    dfy = dfy.iloc[:,:-1]
    dfy[dfy.columns[1]] = dfy[dfy.columns[1]].astype(float)
    dfy

    dfx.to_csv('2017_x_data.csv', index=False)
    dfy.to_csv('2017_y_data.csv', index=False)





