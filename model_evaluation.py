def R2(a, b):
    '''
    :param a: RSS预测残差或MSE
    :param b: TSS总平方差或VAR(y)
    :return:  1-MSE/VAR 或 1-RSS/TSS
    '''
    return 1 - a / b