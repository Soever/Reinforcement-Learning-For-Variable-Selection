import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from data_process import importData,X1PATH,Y1PATH,DataClass

# 定义节点类
class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 当前状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 节点访问次数
        self.value = 0.0  # 节点价值

    def is_fully_expanded(self):
        # 检查此节点是否已完全展开（所有可能的子节点都已被探访）
        return len(self.children) == self.state.num_possible_moves()

    def best_child(self, c_param=1.0):
        # 选择最佳子节点，使用 UCB1 算法
        choices_weights = [
            (c.value / (c.visits + 1e-7)) + c_param * np.sqrt((2 * np.log(self.visits + 1.0)) / (c.visits + 1e-7))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        # 扩展一个新的子节点
        for m in self.state.get_possible_moves():
            # 如果动作 m（即选择特征 m）尚未被尝试，则扩展该节点
            if not any(ch.state.selected_features == self.state.selected_features + [m] for ch in self.children):
                child_state = FeatureSelectionState(self.state.X, self.state.y, self.state.selected_features + [m])
                child = Node(child_state, parent=self)
                self.children.append(child)
                return child


# 定义 MCTS 类
class MCTS:
    def __init__(self, root_state, num_simulations, c_param=1.0):
        self.root = Node(root_state)  # 根节点
        self.num_simulations = num_simulations  # 模拟次数
        self.c_param = c_param  # UCB1 算法中的常数

    def search(self):
        # 主要的搜索函数
        for _ in range(self.num_simulations):
            v = self.tree_policy()  # 根据树策略选择一个节点
            reward = v.state.rollout()  # 进行模拟
            self.backpropagate(v, reward)  # 回传模拟结果
        # 返回最佳子节点
        return self.root.best_child(c_param=0.0)

    def tree_policy(self):
        # 树策略：选择一个待评估的节点
        v = self.root
        while not v.state.is_terminal():
            if not v.is_fully_expanded():
                return v.expand()  # 如果节点没有完全展开，则扩展一个新的子节点
            else:
                v = v.best_child(c_param=self.c_param)  # 如果已完全展开，则选择最佳子节点
        return v

    def backpropagate(self, node, reward):
        # 回传：将模拟得到的奖励传回至根节点
        while node is not None:
            node.visits += 1  # 更新访问次数
            node.value += reward  # 更新价值
            node = node.parent  # 回到父节点


# 定义特征选择状态类
class FeatureSelectionState:
    def __init__(self, X, y, selected_features=None):
        self.X = X  # 特征
        self.y = y  # 目标变量
        self.selected_features = selected_features if selected_features is not None else []  # 已选择的特征列表

    def num_possible_moves(self):
        # 可能的动作数量等于尚未选择的特征数量
        return self.X.shape[1] - len(self.selected_features)

    def get_possible_moves(self):
        # 可能的动作是所有尚未选择的特征
        return [i for i in range(self.X.shape[1]) if i not in self.selected_features]

    def is_terminal(self):
        # 当选择所有特征时，我们达到终止状态
        return len(self.selected_features) == self.X.shape[1]

    def rollout(self):
        # 随机选择一些特征，训练模型，返回负 MSE（因为我们希望最小化 MSE）
        available_features = [i for i in range(self.X.shape[1]) if i not in self.selected_features]
        if not available_features:
            return 0
        num_rollout_features = min(len(available_features),
                                   np.random.randint(1, self.X.shape[1] + 1 - len(self.selected_features)))
        rollout_features = np.random.choice(available_features, size=num_rollout_features, replace=False)
        all_features = self.selected_features + rollout_features.tolist()
        model = DecisionTreeRegressor()
        model.fit(self.X[:, all_features], self.y)
        predictions = model.predict(self.X[:, all_features])
        mse = mean_squared_error(self.y, predictions)
        return -mse


# 生成一些合成数据
# X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
DFcls = DataClass()
Xy =np.array(DFcls.orign_df.dropna())
X,y = Xy[:,:-1],Xy[:,-1]
X = DFcls.scalerX.transform(X)
y = DFcls.scalerY.transform(y.reshape(-1,1))
# 初始化问题状态
initial_state = FeatureSelectionState(X, y)

# 运行 MCTS
tree = MCTS(initial_state, num_simulations=100)
best_child = tree.search()

# 显示最佳特征子集
print(best_child.state.selected_features)
