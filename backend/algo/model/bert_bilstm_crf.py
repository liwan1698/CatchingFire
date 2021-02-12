"""
采用 BERT + BILSTM + CRF 网络进行处理
"""

from algo.model.model_config import BertBilstmCrfConfig

from keras.models import Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras_contrib.layers import CRF
import keras_bert
import os


class BertBilstmCrf(object):
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 max_len: int = 100,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate
        self.config_path = os.path.join(BertBilstmCrfConfig.BERT_MODEL_DIR, 'bert_config.json')
        self.check_point_path = os.path.join(BertBilstmCrfConfig.BERT_MODEL_DIR, 'bert_model.ckpt')
        self.dict_path = os.path.join(BertBilstmCrfConfig.BERT_MODEL_DIR, 'vocab.txt')

    def creat_model(self):
        print('load bert Model start!')
        model = keras_bert.load_trained_model_from_checkpoint(self.config_path,
                                                              checkpoint_file=self.check_point_path,
                                                              seq_len=self.max_len,
                                                              trainable=True)
        print('load bert Model end!')
        inputs = model.inputs
        embedding = model.output
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(embedding)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.n_class)(x)
        self.crf = CRF(self.n_class, sparse_target=False)
        x = self.crf(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(optimizer=Adam(1e-5),
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])


from sklearn.metrics import f1_score, recall_score
import numpy as np
import pandas as pd

from algo.model.utils import *
from keras.callbacks import EarlyStopping
from .process_data import DataProcess

max_len = 100


def train_sample(epochs=15):
    # bert需要不同的数据参数 获取训练和测试数据
    dp = DataProcess(data_type='msra', max_len=max_len, model='bert')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    print("----------------------------数据信息 START--------------------------")
    print(f"当前使用数据集 MSRA")
    # log.info(f"train_data:{train_data.shape}")
    print(f"train_label:{train_label.shape}")
    # log.info(f"test_data:{test_data.shape}")
    print(f"test_label:{test_label.shape}")
    print("----------------------------数据信息 END--------------------------")

    model_class = BertBilstmCrf(dp.vocab_size, dp.tag_size, max_len=max_len)
    model = model_class.creat_model()

    train_model='BERTBILSTMCRF'
    callback = TrainHistory(model_name=train_model)  # 自定义回调 记录训练数据
    early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max')  # 提前结束
    model.fit(train_data, train_label, batch_size=32, epochs=epochs,
              validation_data=[test_data, test_label],
              callbacks=[callback, early_stopping])

    # 计算 f1 和 recall值

    pre = model.predict(test_data)
    pre = np.array(pre)
    test_label = np.array(test_label)
    pre = np.argmax(pre, axis=2)
    test_label = np.argmax(test_label, axis=2)
    pre = pre.reshape(pre.shape[0] * pre.shape[1], )
    test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], )

    f1score = f1_score(pre, test_label, average='macro')
    recall = recall_score(pre, test_label, average='macro')

    print("================================================")
    print(f"--------------:f1: {f1score} --------------")
    print(f"--------------:recall: {recall} --------------")
    print("================================================")

    # 把 f1 和 recall 添加到最后一个记录数据里面
    info_list = callback.info
    if info_list and len(info_list)>0:
        last_info = info_list[-1]
        last_info['f1'] = f1score
        last_info['recall'] = recall

    return info_list


if __name__ == '__main__':
    # columns = ['model_name','epoch', 'loss', 'acc', 'val_loss', 'val_acc', 'f1', 'recall']
    info_list = train_sample(epochs=3)
    for info in info_list:
        print(info)

