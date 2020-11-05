import os  # 文件/目录方法
import feather  # 快速、轻量级的存储框架
import numpy as np  # 大量的维度数组与矩阵运算
import joblib  # numpy数组进行了特定的优化
import pandas as pd  # 大数据处理模块
from tqdm import tqdm  # 显示进度
from sklearn.metrics import f1_score  # 评价指标

#paddle paddle
import paddle
#深度学习框架
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm
from paddle.fluid.layers import dynamic_lstm
from paddle.fluid import layers

strDataDir = 'D:/comp/traffic/'  #数据所在文件夹路径

def data_loader(x_data=None, y_data=None, batch_size=1024):
    ##连续读取feather数据
    #输入x_data为feature -1*5*20  ， y_data为label路况 ，batch_size单次最大读取条目数
    #输出feature列表，label列表
    def reader():
        ##
        batch_data = [] #feature 单元大小5*20
        batch_labels = [] #label路况
        for i in range(x_data.shape[0]):
            batch_labels.append(y_data[i])
            batch_data.append(x_data[i])
            if len(batch_data) == batch_size:
                batch_data = np.array(batch_data).astype('float32')
                batch_labels = np.array(batch_labels).astype('int')
                yield batch_data, batch_labels
                #带有yield的函数是一个迭代器，函数返回某个值时，会停留在某个位置，返回函数值后，会在前面停留的位置继续执行，直到程序结束；return返回的是一个list列表，而yield每次调用只返回一个数值，毫无疑问，使用return空间开销比较大，尤其是操作巨量数据的时候，操作一个大列表时间开销也会得不偿失
                batch_data = []
                batch_labels = []
        if len(batch_data) > 0:
            batch_data = np.array(batch_data).astype('float32')
            batch_labels = np.array(batch_labels).astype('int')
            yield batch_data, batch_labels
            batch_data = []
            batch_labels = []

    return reader

##训练模型
class base_model(fluid.dygraph.Layer):
    def __init__(self, classes_num: int):
        ##初始化函数
        super().__init__()
        self.hidden_size = 128
        self.batchNorm1d = paddle.nn.BatchNorm1D(5)  # BatchNorm1d(5)
        self.lstm = paddle.nn.LSTM(
            input_size=20, hidden_size=self.hidden_size, direction="bidirectional")
        self.avgpool1d = paddle.nn.AvgPool1D(  # AvgPool1d
            kernel_size=self.hidden_size*2, stride=self.hidden_size*2)
        self.maxpool1d = paddle.nn.MaxPool1D(  # MaxPool1d
            kernel_size=self.hidden_size*2, stride=self.hidden_size*2)

    def forward(self, input):
        #input:（batch_size, max_len, dim)
        x = self.batchNorm1d(input)
        x.stop_gradient = True

        rnn_out = self.lstm(x)[0]
        mean_out = self.avgpool1d(rnn_out)
        max_out = self.maxpool1d(rnn_out)
        r_shape = (mean_out.shape[0], mean_out.shape[1])
        mean_pool_out = layers.reshape(mean_out, shape=r_shape)
        max_pool_out = layers.reshape(max_out, shape=r_shape)
        #add_output = mean_pool_out + max_pool_out
        concat_output = layers.concat((mean_pool_out, max_pool_out), axis=1)
        output = layers.fc(concat_output, size=5)
        return output

#__name__是python的一个内置类属性，它天生就存在于一个 python 程序中，代表对应程序名称。
if __name__ == '__main__':
    # 创建模型
    # with fluid.dygraph.guard():
    program = fluid.default_main_program()
    program.random_seed = 2020
    model = base_model(4)
    print('start training ... {} kind'.format(4))
    model.train()
    epoch_num = 30
    # 定义优化器
    opt = fluid.optimizer.Adam(
        learning_rate=0.001, parameter_list=model.parameters())
    # 定义数据读取器，训练数据读取器和验证数据读取器
    x = joblib.load(strDataDir+'20190701_x.pkl')
    y = joblib.load(strDataDir+'20190701_y.pkl')
    val_x = joblib.load(strDataDir+'20190702_x.pkl')
    val_y = joblib.load(strDataDir+'20190702_y.pkl')
    train_loader = data_loader(x, y, 1024)
    valid_loader = data_loader(val_x, val_y, 1024)

    best_acc = 0
    valid_acc = 0

    print('start training ... {} kind'.format(4))
    for epoch in range(epoch_num):
        all_loss = 0
        model.train()

        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            x = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            label = paddle.fluid.one_hot(label, depth=5)
            # 运行模型前向计算，得到预测值
            logits = model(x)
            # 进行loss计算
            softmax_logits = fluid.layers.softmax(logits)
            loss = fluid.layers.cross_entropy(
                softmax_logits, label, soft_label=True)
            avg_loss = fluid.layers.mean(loss)
            all_loss += avg_loss.numpy()
            avg_l = all_loss/(batch_id + 1)
            if(batch_id % 100 == 0):
                print("epoch: {}, batch_id: {}, loss is: {}, valid acc is: {}".format(
                    epoch, batch_id, avg_loss.numpy(), valid_acc))
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
            # break
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            x_data = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(x_data)
            # 计算sigmoid后的预测概率，进行loss计算
            pred = fluid.layers.softmax(logits)

            scores = f1_score(y_true=pred.numpy().argmax(
                axis=1), y_pred=y_data, average=None)

            scores = scores[0]*0.2 + scores[1]*0.2 + scores[2]*0.6
            accuracies.append(scores)
        valid_acc = np.mean(accuracies)
