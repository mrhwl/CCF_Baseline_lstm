# !pip install -U feather-format
# !python3 -m pip install paddlepaddle-gpu==2.0.0b0 -i https://mirror.baidu.com/pypi/simple
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

#paddle.disable_static() #切换到动态图模式 2.0-rc版本可以删除这一行，因为默认为这模式
print(paddle.__version__)  # 显示版本
strDataDir = 'D:/comp/traffic/'  #数据所在文件夹路径

def txt2feather(path):
    ## 将txt文件转为feather文件，读取速度比较快
    ## eg:
    ## for p in tqdm(os.listdir('train/traffic/')):
    ## txt2feather(p)
    df = pd.read_csv(f'{strDataDir+path}', sep=';', names=['linkid_' 'label_' 'current_slice_id_' 'future_slice_id',
                                                                'recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4'])#-1*6
    df_test = pd.DataFrame(
        columns=['linkid', 'label', 'current_slice_id', 'future_slice_id'])#-1*4
    df_test[['linkid', 'label', 'current_slice_id', 'future_slice_id']
            ] = df.iloc[:, 0].str.split(' ', expand=True) #iloc的用法 iloc[ : , : ] 前面的冒号就是取行数,后面的冒号是取列数
    df_test = df_test.astype('int') #类型转换
    df_test[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']] = df[[
        'recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']]#-1*5
    df_test.to_feather(f'{strDataDir+path[:-4]}.feather')#-4表示负索引，从左截取到倒数第4个字符的位置，去除“.txt”
    return df_test#-1*9，共9个列名


def get_feature(df):
    ##将recent_feature, history_feature1-4,合并并且储存成lstm所能使用格式。
    ##输入df -1*9
    ##输出lable(路况)，以及(-1, 5, 20)的numpy数组。
    def get_split(df_t):
        #输入每个feature块数组，整理数据
        arr_list = []
        print('start a new split, please wait patiently, it takes about two minues to get done')
        for i in range(5):
            f = df_t[i].str.split(':', expand=True)[1].str.split(
                ',', expand=True).astype(float).values.reshape((-1, 1, 4))#numpy 转换成1*1*4矩阵 目的组成时序
            arr_list.append(f)
        return np.concatenate(arr_list, axis=1)#numpy 数组拼接 axis=0表示层 axis=1表示行 axis=2列 最后返回1*5*4矩阵块

    recent_feature = df.recent_feature.str.split(' ', expand=True)#分割feature块转存为数组，大小为4
    recent_feature = get_split(recent_feature)#获取1*5*4格式的数组

    history_feature1 = df.history_feature1.str.split(' ', expand=True)
    history_feature1 = get_split(history_feature1)

    history_feature2 = df.history_feature2.str.split(' ', expand=True)
    history_feature2 = get_split(history_feature2)

    history_feature3 = df.history_feature3.str.split(' ', expand=True)
    history_feature3 = get_split(history_feature3)

    history_feature4 = df.history_feature4.str.split(' ', expand=True)
    history_feature4 = get_split(history_feature4)

    combined_feature = np.concatenate(
        [recent_feature, history_feature1, history_feature2, history_feature3, history_feature4], axis=2)#1*5*20

    return combined_feature, df.label


##将test集转化为lstm输入格式
# df_0702 = pd.read_csv('test.txt', sep=';',names=['linkid_' 'label_' 'current_slice_id_' 'future_slice_id',
#                              'recent_feature', 'history_feature1', 'history_feature2','history_feature3', 'history_feature4'])
# df_test = pd.DataFrame(
#     columns=['linkid', 'label', 'current_slice_id', 'future_slice_id'])
# df_test[['linkid', 'label', 'current_slice_id', 'future_slice_id']
#         ] = df_0702.iloc[:, 0].str.split(' ', expand=True)
# df_test = df_test.astype('int')
# df_test[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']
#         ] = df_0702[['recent_feature', 'history_feature1', 'history_feature2', 'history_feature3', 'history_feature4']]
# df_test.to_feather('test.feather')
# test = feather.read_dataframe('test.feather')
# test=txt2feather('test.txt')#上面可以用这个代替吧
# test_x, test_y = get_feature(test)
# joblib.dump(test_x, strDataDir+'test_x.pkl')#feature 存储数据
# joblib.dump(test_y, strDataDir+'test_y.pkl')#label
##将train集中7月1日当成训练集，7月2日当成验证集，转化为lstm输入格式
train_data = txt2feather('20190701.txt')
train_x, train_y = get_feature(train_data)
joblib.dump(train_x, strDataDir+'20190701_x.pkl')
joblib.dump(train_y, strDataDir+'20190701_y.pkl')
##验证集
valid_data = txt2feather('20190702.txt')
valid_x, valid_y = get_feature(valid_data)
joblib.dump(valid_x, strDataDir+'20190702_x.pkl')
joblib.dump(valid_y, strDataDir+'20190702_y.pkl')
