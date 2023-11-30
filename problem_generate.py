import  numpy as np

class Problem:
    def __init__(self,feature_num,lb,ub,resolution=1):
        self.feature_num = feature_num
        self.lb = lb
        self.ub = ub

    def generate_action_space(self):
        pass
