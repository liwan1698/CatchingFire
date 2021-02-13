"""
采用 BERT + BILSTM + CRF 网络进行处理
"""
import json

import jieba
from django.db.models import Q

from algo.model.model import CustomModel
from algo.model.model_config import BertBilstmCrfConfig
from keras.models import Model
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras_contrib.layers import CRF
import keras_bert
import os

from algo.models import NerData


# 获取词典
unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'

class BertBilstmCrf(CustomModel):
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
        self.epochs = 15
        self.w2i = get_w2i()  # word to index
        self.one_hot = True
        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)
        self.tag2index = get_tag2index()  # tag to index
        self.tag2index = get_tag2index()  # tag to index
        self.tag_size = len(self.tag2index)

    def precess_data(self):
        # 从数据库读取
        queryset = NerData.objects.filter(~Q(human_tag=None))
        # poses=[{"begin": 2, "end": 3, "pos": "LOC"}]
        sentences = []
        tags = []
        for q in queryset:
            sentence = q['text']
            poses = json.loads(q['human_label'])
            # 整理标注数据
            tag = ['O'] * len(sentence)
            for pos in poses:
                begin = int(pos['begin'])
                end = int(pos['end'])
                pos_tag = pos['pos']
                tag[begin] = f"B-{pos_tag}"
                if end > begin:
                    tag[begin+1:end] = (end-begin-1) * [f"I-{pos_tag}"]
            tags.append(tag)
            sentences.append(sentence)
        # 转化
        data = self.data_to_index(sentences)
        label = self.label_to_index(tags)

        # 进行 one-hot处理
        if self.one_hot:
            def label_to_one_hot(index: []) -> []:
                data = []
                for line in index:
                    data_line = []
                    for i, index in enumerate(line):
                        line_line = [0]*self.tag_size
                        line_line[index] = 1
                        data_line.append(line_line)
                    data.append(data_line)
                return np.array(data)
            data_label = label_to_one_hot(index=label)
        else:
            data_label = np.expand_dims(label, 2)
        train_data_proportion = 0.8
        num = len(data[0])
        self.train_data = [data[0][:, int(train_data_proportion*num):], data[1][:, int(train_data_proportion*num):]]
        self.train_label = data_label[:, int(train_data_proportion*num):]
        self.test_data = [data[0][:, :int(train_data_proportion*num)], data[1][:, :int(train_data_proportion*num)]]
        self.test_label = data_label[:, :int(train_data_proportion*num)]


    def label_to_index(self, tags):
        """
        将训练数据x转化为index
        :return:
        """
        label_ids = []
        line_label = []
        for tag in tags:
            for t in tag:
                # bert 需要输入index和types 由于我们这边都是只有一句的，所以type都为0
                t_index = self.tag2index.get(t, 0)
                line_label.append(t_index)  # label index

            max_len_buff = self.max_len-2
            if len(line_label) > max_len_buff: # 先进行截断
                line_label = line_label[:max_len_buff]
            line_label = [0] + line_label + [0]

            # padding
            if len(line_label) < self.max_len: # 填充到最大长度
                pad_num = self.max_len - len(line_label)
                line_label = [0] * pad_num + line_label
            label_ids.append(np.array(line_label))
            line_label = []
        return np.array(label_ids)

    def data_to_index(self, sentences):
        """
        将训练数据x转化为index
        :return:
        """
        data_ids = []
        data_types = []
        line_data_ids = []
        line_data_types = []
        for sentence in sentences:
            for w in sentence:
                # bert 需要输入index和types 由于我们这边都是只有一句的，所以type都为0
                w_index = self.w2i.get(w, self.unk_index)
                line_data_ids.append(w_index)  # index
                line_data_types.append(0)  # types

            max_len_buff = self.max_len-2
            if len(line_data_ids) > max_len_buff: # 先进行截断
                line_data_ids = line_data_ids[:max_len_buff]
                line_data_types = line_data_types[:max_len_buff]
            line_data_ids = [self.cls_index] + line_data_ids + [self.sep_index]
            line_data_types = [0] + line_data_types + [0]

            # padding
            if len(line_data_ids) < self.max_len: # 填充到最大长度
                pad_num = self.max_len - len(line_data_ids)
                line_data_ids = [self.pad_index]*pad_num + line_data_ids
                line_data_types = [0] * pad_num + line_data_types
            data_ids.append(np.array(line_data_ids))
            data_types.append(np.array(line_data_types))
            line_data_ids = []
            line_data_types = []
        return [np.array(data_ids), np.array(data_types)]

    def build(self):
        self.precess_data()
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
        self.model.compile(optimizer=Adam(1e-5),
                           loss=self.crf.loss_function,
                           metrics=[self.crf.accuracy])

    def train(self):
        train_model='BERTBILSTMCRF'
        callback = TrainHistory(model_name=train_model)  # 自定义回调 记录训练数据
        early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max')  # 提前结束
        self.model.fit(self.train_data, self.train_label, batch_size=32, epochs=self.epochs,
                  validation_data=[self.test_data, self.test_label],
                  callbacks=[callback, early_stopping])

        # 计算 f1 和 recall值
        pre = self.model.predict(self.test_data)
        pre = np.array(pre)
        test_label = np.array(self.test_label)
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

    def predict_all(self):
        # 预测所有数据，并存储到数据库
        queryset = NerData.objects.all()
        sentences = [s['text'] for s in queryset]
        data = self.data_to_index(sentences)
        predict = self.model.predict(data)
        # todo 将预测结果存储到数据库


from sklearn.metrics import f1_score, recall_score
import numpy as np
from keras.callbacks import EarlyStopping
from algo.model.process_data import DataProcess, get_w2i, get_tag2index

max_len = 100


def train_sample(epochs=15):
    # bert需要不同的数据参数 获取训练和测试数据
    dp = DataProcess(data_type='msra', max_len=max_len, model='bert')
    # todo 改为从数据库读取数据
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    print("----------------------------数据信息 START--------------------------")
    print(f"当前使用数据集 MSRA")
    # log.info(f"train_data:{train_data.shape}")
    print(f"train_label:{train_label.shape}")
    # log.info(f"test_data:{test_data.shape}")
    print(f"test_label:{test_label.shape}")
    print("----------------------------数据信息 END--------------------------")

    model_class = BertBilstmCrf(dp.vocab_size, dp.tag_size, max_len=max_len)
    model = model_class.build()

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


import keras


class TrainHistory(keras.callbacks.Callback):
    def __init__(self, model_name=None):
        super(TrainHistory, self).__init__()
        self.model_name = model_name
        self.epoch = 0
        self.info = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        message = f"begin epoch: {self.epoch}"
        print(message)

    def on_epoch_end(self, epoch, logs={}):
        message = f'end epoch: {epoch} loss:{logs["loss"]} val_loss:{logs["val_loss"]} acc:{logs["crf_viterbi_accuracy"]} val_acc:{logs["val_crf_viterbi_accuracy"]}'
        print(message)
        dict = {
            'model_name':self.model_name,
            'epoch': self.epoch+1,
            'loss': logs["loss"],
            'acc': logs['crf_viterbi_accuracy'],
            'val_loss': logs["val_loss"],
            'val_acc': logs['val_crf_viterbi_accuracy']
        }
        self.info.append(dict)

    def on_batch_end(self, batch, logs={}):
        message = f'{self.model_name} epoch: {self.epoch} batch:{batch} loss:{logs["loss"]}  acc:{logs["crf_viterbi_accuracy"]}'
        print(message)

