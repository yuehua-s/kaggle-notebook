# Kaggle competition url:
# https://www.kaggle.com/competitions/titanic/code

import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# Import Data
train_data = pd.read_csv("../kaggle_data/titanic/train.csv")
test_data = pd.read_csv("../kaggle_data/titanic/test.csv")

# Data Preprocessing
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
n_train = train_data.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32, device=d2l.try_gpu())
train_labels = torch.tensor(train_data.Survived.values.reshape(-1, 1), dtype=torch.long, device=d2l.try_gpu())
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32, device=d2l.try_gpu())

# Neural Network Model(MLP+SoftMax)
in_features = train_features.shape[1]
def get_net():
    return nn.Sequential(nn.Linear(in_features, 4800), nn.ReLU(),
                         nn.Linear(4800, 2400), nn.ReLU(),
                         nn.Linear(2400, 1200), nn.ReLU(),
                         nn.Linear(1200, 600), nn.ReLU(),
                         nn.Linear(600, 300), nn.ReLU(),
                         nn.Linear(300, 100), nn.ReLU(),
                         nn.Linear(100, 2))

# Evaluation Model
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter, loss):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    # 关闭net自动求导
    with torch.no_grad():
        for X, y in data_iter:
            y = y.squeeze(1)
            y_hat = net(X)
            l = loss(y_hat, y)
            metric.add(float(l.mean()), accuracy(y_hat, y), y.numel())
    # 返回测试损失和测试精度
    return metric[0] / metric[2], metric[1] / metric[2]

# Train
def train_epoch(net, train_iter, loss, optimizer):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(xavier_init_weights)
    for X, y in train_iter:
        optimizer.zero_grad()
        # y的维度从([20, 1])降至([20])
        y = y.squeeze(1)
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        optimizer.step()
        metric.add(float(l.mean()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.05],
                            legend=['train loss', 'train acc', 'train loss', 'test acc'])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    if test_labels is not None:
        test_iter = d2l.load_array((test_features, test_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    loss = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, optimizer)
        if test_labels is not None:
            test_metrics = evaluate_accuracy(net, test_iter, loss)
            animator.add(epoch + 1, train_metrics + test_metrics)
        else:
            animator.add(epoch + 1, train_metrics)

# K折交叉训练验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
def k_fold(k, train_features, train_labels, num_epochs,
           learning_rate, weight_decay, batch_size, device):
    for i in range(k):
        print(f'折{i + 1}')
        data = get_k_fold_data(k, i, train_features, train_labels)
        net = get_net()
        net.to(device)
        train(net, *data,
              num_epochs, learning_rate, weight_decay, batch_size)

# Submit Predictions
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, learning_rate, weight_decay, batch_size, device):
    """选好合适的网络后，拿全部的数据重新训练、预测并生成提交数据"""
    net = get_net()
    net.to(device)
    train(net, train_features, train_labels, None, None,
          num_epochs, learning_rate, weight_decay, batch_size)
    # 将网络应用于测试集。
    preds = net(test_features).argmax(axis=1).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['Survived'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['PassengerId'], test_data['Survived']], axis=1)
    submission.to_csv('/Users/zhangyuehua/Desktop/titanic_submission.csv', index=False)

# Main
if __name__ == '__main__':
    """合理定义超参数并开始炼丹"""
    k, num_epochs, batch_size, learning_rate, weight_decay, device = 5, 15, 50, 0.003, 0, d2l.try_gpu()
    k_fold(k, train_features, train_labels, num_epochs,
           learning_rate, weight_decay, batch_size, device)
    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, learning_rate, weight_decay, batch_size, device)
    d2l.plt.show()