# Kaggle competition url:
# https://www.kaggle.com/competitions/forest-cover-type-prediction

# %matplotlib inline
import math
import time
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# Import Data
train_data = pd.read_csv("../../kaggle_data/forestCoverTypePrediction/train.csv")
test_data = pd.read_csv("../../kaggle_data/forestCoverTypePrediction/test.csv")

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
train_labels = torch.tensor(train_data.Cover_Type.values.reshape(-1, 1), dtype=torch.long, device=d2l.try_gpu())
train_labels = train_labels - 1 # label从1-7，nn.CrossEntropyLoss需要的是0-6
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32, device=d2l.try_gpu())
print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)

# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# ffn = PositionWiseFFN(648, 1296, 648)
# ffn.eval()
# # X是[batch_size, num_steps, num_features]
# X = torch.ones((64, 1, 648))
# print(ffn(X).shape)

# 残差连接和层规范化
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        # self.ln = nn.LayerNorm(normalized_shape)
        self.bn = nn.BatchNorm1d(normalized_shape)
    def forward(self, X, Y):
        return self.bn(self.dropout(Y) + X) # [batch_size, num_hiddens, num_steps]

# add_norm = AddNorm(648, 0.1)
# add_norm.eval()
# # [batch_size, num_features, num_steps]
# X = torch.ones((64, 648, 1))
# res_add_norm = add_norm(X, X)
# print("res_add_norm.shape", res_add_norm.shape)

# 编码器
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        print("EncoderBlock input X.shape", X.shape)

        X_mha = self.attention(X, X, X, valid_lens)
        # 将X的形状调整为[batch_size, num_features, num_steps]
        X = X.permute(0, 2, 1)
        X_mha = X_mha.permute(0, 2, 1)
        # print("n1 X.shape", X.shape)
        # print("n1 X_mha.shape", X_mha.shape)

        Y = self.addnorm1(X, X_mha)
        # print("Y.shape", Y.shape)
        Y = Y.permute(0, 2, 1)
        # print("Y.shape", Y.shape)

        Y_ffn = self.ffn(Y)
        # print("Y_ffn.shape", Y_ffn.shape)
        Y = Y.permute(0, 2, 1)
        Y_ffn = Y_ffn.permute(0, 2, 1)
        # print("n2 Y.shape", Y.shape)
        # print("n2 Y_ffn.shape", Y_ffn.shape)

        # print("EncoderBlock output Y.shape", self.addnorm2(Y, Y_ffn).shape)
        return self.addnorm2(Y, Y_ffn).permute(0, 2, 1)

# # [batch_size, num_steps, num_features]
# X = torch.ones((64, 1, 648))
# valid_lens = torch.ones([64])
# encoder_blk = EncoderBlock(key_size=648, query_size=648, value_size=648, num_hiddens=648,
#                            norm_shape=([648]), ffn_num_input=648, ffn_num_hiddens=1296, num_heads=8,
#                            dropout=0.1)
# encoder_blk.eval()
# res = encoder_blk(X, valid_lens)
# print("res.shape", res.shape)
# time.sleep(3600)


class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)

        self.num_hiddens = num_hiddens
        # 这里因为我们的输入train_iter已经是[batch_size, num_steps, features]，
        # 所以不需要再Embedding了，而是通过一个Linear实现相同功能。
        # self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.linear = nn.Linear(vocab_size, num_hiddens)

        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))
    def forward(self, X, valid_lens, *args):
        print("TransformerEncoder Input X.shape", X.shape)
        print("TransformerEncoder Input valid_lens.shape", valid_lens.shape)
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.linear(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        # X.shape torch.Size([64, 1, 648])[batch_size, num_steps, num_features]
        return X


class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, num_hiddens, num_label_features, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.dense = nn.Sequential(nn.Linear(num_hiddens, 4096), nn.ReLU(),
                                  nn.Linear(4096, 4096), nn.ReLU(),
                                  nn.Linear(4096, 1000), nn.ReLU(),
                                  nn.Linear(1000, num_label_features))
            # nn.Linear(num_hiddens, num_label_features)
    def forward(self, state):
        return self.dense(state)

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, enc_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        return self.decoder(enc_outputs)

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    print("accuracy y", y)
    print("accuracy y_hat.type(y.dtype)", y_hat.type(y.dtype))
    cmp = y_hat.type(y.dtype) == y
    print("accuracy cmp", cmp)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, loss):
    """计算在指定数据集上模型的精度"""
    net.eval()  # 将模型设置为评估模式
    net.to(device)
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    # 关闭net自动求导
    with torch.no_grad():
        for X, y in data_iter:
            X = X.unsqueeze(1)  # 增加num_steps维度
            # print("evaluate_accuracy X.shape:", X.shape)
            valid_lens = torch.ones(X.shape[0]).to(device)
            # print("evaluate_accuracy valid_lens.shape", valid_lens.shape)
            # y的维度从([batch_size, 1])降至([batch_size])
            y = y.squeeze(dim=1)
            y_hat = net(X, valid_lens).squeeze(dim=1)  # [batch_size, num_steps, num_features] -> [batch_size, num_features]
            print("evaluate_accuracy y.shape", y.shape)
            print("evaluate_accuracy y_hat.shape", y_hat.shape)
            l = loss(y_hat, y)
            # metric.add(float(l.mean()), accuracy(y_hat, y), y.numel())
            metric.add(float(l.mean()), accuracy(y_hat, y), y.numel())
    # 返回测试损失和测试精度
    return metric[0] / metric[2], metric[1] / metric[2]

# Train
def train_epoch(net, train_iter, loss, optimizer):
    net.train()  # 将模型设置为训练模式
    net.to(device)
    # 训练损失总和、训练准确度总和、样本数
    metric = d2l.Accumulator(3)
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    for X, y in train_iter:
        optimizer.zero_grad()
        # X [batch_size, num_features] -> [batch_size, num_steps, num_features]
        X = X.unsqueeze(1)  # 增加num_steps维度
        # print("train_epoch X.shape:", X.shape)
        valid_lens = torch.ones(X.shape[0]).to(device)
        # print("train_epoch valid_lens.shape", valid_lens.shape)
        # y的维度从([batch_size, 1])降至([batch_size])
        y = y.squeeze(dim=1)
        y_hat = net(X, valid_lens).squeeze(dim=1) # [batch_size, num_steps, num_features] -> [batch_size, num_features]
        print("train_epoch y.shape", y.shape)
        print("train_epoch y_hat.shape", y_hat.shape)
        l = loss(y_hat, y)
        # l.mean().backward()
        l.sum().backward()
        optimizer.step()
        metric.add(float(l.mean()), accuracy(y_hat, y), y.numel())
        # metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 3],
                            legend=['train loss', 'train acc', 'test loss', 'test acc'])
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

def k_fold(net, k, train_features, train_labels, num_epochs,
           learning_rate, weight_decay, batch_size, device):
    for i in range(k):
        print(f'折{i + 1}')
        data = get_k_fold_data(k, i, train_features, train_labels)
        net.to(device)
        train(net, *data,
              num_epochs, learning_rate, weight_decay, batch_size)


# def train_and_pred(net, train_features, test_features, train_labels, test_data, valid_lens,
#                    num_epochs, learning_rate, weight_decay, batch_size, device):
#     """选好合适的网络后，拿全部的数据重新训练、预测并生成提交数据"""
#     net.to(device)
#     train(net=net, train_features=train_features, train_labels=train_labels, test_features=None, test_labels=None,
#           num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)
#
#     # 将网络应用于测试集。
#     preds = net(test_features, valid_lens).argmax(axis=1).detach().numpy() + 1 # +1是因为之前把Cover_Type都减去1做的训练，结果再加回来。
#     # 将其重新格式化以导出到Kaggle
#     test_data['Cover_Type'] = pd.Series(preds.reshape(1, -1)[0])
#     submission = pd.concat([test_data['Id'], test_data['Cover_Type']], axis=1)
#     submission.to_csv('../../kaggle_data/forestCoverTypePrediction/forest_over_type_prediction_submission.csv', index=False)

if __name__ == '__main__':
    """合理定义超参数并开始炼丹"""
    num_layers, dropout, batch_size, num_steps = 20, 0, 200, 1
    learning_rate, num_epochs, device = 5, 100, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 1080, 2160, 8
    key_size = query_size = value_size = num_hiddens = 1080
    norm_shape = ([1080])
    num_features = 54
    num_label_features = 7 # label是从1-7，而不是0-6（index非从0开始，7只能从0-6），做nn.CrossEntropyLoss会报错。参考：https://discuss.pytorch.org/t/indexerror-target-is-out-of-bounds/84417
    k, weight_decay = 5, 0
    encoder = TransformerEncoder(vocab_size=num_features, key_size=key_size, query_size=query_size, value_size=value_size,
                                 num_hiddens=num_hiddens, norm_shape=norm_shape, ffn_num_input=ffn_num_input, ffn_num_hiddens=ffn_num_hiddens,
                                 num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    decoder = TransformerDecoder(num_hiddens=num_hiddens, num_label_features=num_label_features)
    net = EncoderDecoder(encoder, decoder)
    k_fold(net=net, k=k, train_features=train_features, train_labels=train_labels, num_epochs=num_epochs,
           learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, device=device)
    # train_and_pred(net=net, train_features=train_features, test_features=test_features, train_labels=train_labels, test_data=test_data, valid_lens=valid_lens,
    #                num_epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size, device=device)

    print("FINISHED")