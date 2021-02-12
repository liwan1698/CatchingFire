"""
使用fasttext做文本分类
"""
from algo.models import ClassifyData
from .model_config import FastTextConfig, MODEL_SAVE_PATH
from .model import Model
# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.global_variables_initializer()
import os
import numpy as np
import logging
from tqdm import tqdm
import sklearn.metrics as metrics
from algo.model.process_data import pre_process, load_dataset, build_vocab, read_vocab, \
    build_label, \
    read_label, \
    data_transform, creat_batch_data, text_processing, read_label_db

from django.db.models import F

# GPU配置信息
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"                  # 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                        # 设置当前使用的GPU设备仅为0号设备
gpuConfig = tf.ConfigProto()
gpuConfig.allow_soft_placement = True                           #设置为True，当GPU不存在或者程序中出现GPU不能运行的代码时，自动切换到CPU运行
gpuConfig.gpu_options.allow_growth = True                       #设置为True，程序运行时，会根据程序所需GPU显存情况，分配最小的资源
gpuConfig.gpu_options.per_process_gpu_memory_fraction = 0.8     #程序运行的时，所需的GPU显存资源最大不允许超过rate的设定值


class FastText(Model):
    def __init__(self, data_index=None, realtime_train=False):
        super().__init__()
        # 配置参数
        self.realtime_train = realtime_train
        self.data_index = data_index    # 数据库里数据的索引开始和结束，例如[0, 100]
        self.input_x = tf.placeholder(shape=[None, FastTextConfig.SEQ_LENGTH],
                                                dtype=tf.int32, name='input-x')      # 输入文本
        self.input_y = tf.placeholder(shape=[None, FastTextConfig.NUM_CLASSES],
                                                dtype=tf.int32, name='input-y')     # 输入文本对应的true label
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, name='input-keep-prob')                     # keep-prob
        self.sess = tf.Session(config=gpuConfig)
        # 加载停用词
        self.stopwords = [word.replace('\n', '').strip() for word in open(FastTextConfig.STOP_WORDS_PATH,
                                                                          encoding='UTF-8')]
        self.optimizer = None
        self.output = None
        self.loss = None
        self.accuracy = None

    def build(self):
        """
        构建fasttext模型，包括embedding层，平均层，全连接层（分层softmax）
        :return:
        """
        # Embedding layer
        embedding = tf.get_variable(shape=[FastTextConfig.VOCAB_SIZE, FastTextConfig.EMBEDDING_DIM],
                                              dtype=tf.float32, name='embedding')
        embedding_x = tf.nn.embedding_lookup(params=embedding, ids=self.input_x)        # dim:(batch_size, 100, 300)

        # dropout层   对表征后的词向量进行dropout操作
        dropout_x = tf.layers.dropout(inputs=embedding_x, rate=self.input_keep_prob)
        # 对文本中的词向量进行求和平均
        embedding_mean_x = tf.reduce_mean(input_tensor=dropout_x, axis=1)     # dim:(batch_size, 300)

        # 全连接层，后接dropout及relu
        fc = tf.layers.dense(inputs=embedding_mean_x, units=128, name='fc1')
        fc = tf.nn.relu(fc)
        # 输出层
        logits = tf.layers.dense(inputs=fc, units=FastTextConfig.NUM_CLASSES, name='logits')
        self.output = tf.argmax(input=tf.nn.softmax(logits), axis=1, name='predict')        # 预测类别
        # 损失函数，交叉熵
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits)
        self.loss = tf.reduce_mean(cross_entropy)
        # 优化器
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FastTextConfig.LEARNING_RATE).minimize(
            loss=self.loss)
        self.sess.run(tf.global_variables_initializer())
        # 准确率
        correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.output)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype=tf.float32))

    def train(self):
        # 数据集预处理
        # if not os.path.exists(FastTextConfig.PREPROCESS_PATH):
        #     pre_process(FastTextConfig.ORIGINAL_DATA_PATH, FastTextConfig.PREPROCESS_PATH)

        # sentences, labels = load_dataset(FastTextConfig.PREPROCESS_PATH)      # 加载数据集
        # if self.realtime_train:
        #     data = ClassifyData.objects.values("pre_text", "human_tag").filter()
            # data = ClassifyData.objects.values("pre_text", "human_tag").filter(predict_tag__ne=F('human_tag'))
        # else:
        #     data = ClassifyData.objects.values("pre_text", "human_tag").exclude(human_tag=None)
        # sentences, labels = [d["pre_text"] for d in data], [d["human_tag"] for d in data]
        # 构建词汇映射表
        # if not os.path.exists(FastTextConfig.VOCAB_PATH):
        #     build_vocab(sentences, FastTextConfig.VOCAB_PATH)
        # word_to_id = read_vocab(FastTextConfig.VOCAB_PATH)      # 读取词汇表及其映射关系
        # 构建类别映射表
        # if not os.path.exists(FastTextConfig.LABEL_PATH):
        #     build_label(labels, FastTextConfig.LABEL_PATH)
        # label_to_id = read_label_db()     # 读取类别表及其映射关系

        # 构建训练数据集
        # data_sentences, data_labels = data_transform(sentences, labels, word_to_id, label_to_id)

        # 构建训练数据集，如果是实时训练，则获取前100条数据，如果离线，则获取全量有效数据
        if self.realtime_train:
            data = ClassifyData.objects.values("pre_text", "human_tag").filter()
        else:
            data = ClassifyData.objects.values("pre_text", "human_tag").exclude(human_tag=None)
        data_sentences = [d['pre_text'] for d in data]
        data_labels = [d['human_tag'] for d in data]

        # 训练集、测试集划分
        split_index = int(len(data_sentences) * FastTextConfig.TRAIN_TEST_SPLIT_VALUE)
        train_data, test_data = data_sentences[: split_index], data_sentences[split_index: ]
        train_label, test_label = data_labels[: split_index], data_labels[split_index: ]

        # 打印训练、测试数据量，数据与标签量是否相等
        logging.info('Train Data: {}'.format(np.array(train_data).shape))
        logging.info('Train Label: {}'.format(np.array(train_label).shape))
        logging.info('Test Data: {}'.format(np.array(test_data).shape))
        logging.info('Test Label: {}'.format(np.array(test_label).shape))

        # 配置Saver
        saver = tf.train.Saver()
        if not os.path.exists(FastTextConfig.MODEL_SAVE_PATH):      # 如不存在相应文件夹，则创建
            os.mkdir(FastTextConfig.MODEL_SAVE_PATH)
        # 是否实时训练
        if self.realtime_train:
            saver.restore(sess=self.sess, save_path=FastTextConfig.MODEL_SAVE_PATH)
        # 模型训练
        best_f1_score = 0  # 初始best模型的F1值
        for epoch in range(1, FastTextConfig.EPOCHS + 1):
            train_accuracy_list = []    # 存储每个epoch的accuracy
            train_loss_list = []        # 存储每个epoch的loss
            # 将训练数据进行 batch_size 切分
            batch_train_data, batch_train_label = creat_batch_data(train_data, train_label, FastTextConfig.BATCH_SIZE)
            for step, (batch_x, batch_y) in tqdm(enumerate(zip(batch_train_data, batch_train_label))):
                feed_dict = {self.input_x: batch_x,
                             self.input_y: batch_y,
                             self.input_keep_prob: FastTextConfig.KEEP_PROB}
                train_accuracy, train_loss, _ = self.sess.run([self.accuracy, self.loss, self.optimizer], feed_dict=feed_dict)
                train_accuracy_list.append(train_accuracy)
                train_loss_list.append(train_loss)
                # 完成一个epoch的训练，输出训练数据的mean accuracy、mean loss
            logging.info('Train Epoch: %d , Loss: %.6f , Acc: %.6f' % (epoch,
                                                                       float(np.mean(np.array(train_loss_list))),
                                                                       float(np.mean(np.array(train_accuracy_list)))))
            # 模型验证
            test_accuracy_list = []         # 存储每个epoch的accuracy
            test_loss_list = []             # 存储每个epoch的loss
            test_label_list = []            # 存储数据的true label
            test_predictions = []           # 存储模型预测出的label
            batch_test_data, batch_test_label = creat_batch_data(test_data, test_label, FastTextConfig.BATCH_SIZE)
            for (batch_x, batch_y) in tqdm(zip(batch_test_data, batch_test_label)):
                feed_dict = {self.input_x: batch_x,
                             self.input_y: batch_y,
                             self.input_keep_prob: 1.0}
                test_predict, test_accuracy, test_loss = self.sess.run([self.output, self.accuracy, self.loss], feed_dict=feed_dict)
                test_accuracy_list.append(test_accuracy)
                test_loss_list.append(test_loss)
                test_label_list.extend(batch_y)
                test_predictions.extend(test_predict)

            # 获取最大score所在的index
            true_y = [np.argmax(label) for label in test_label_list]
            # 计算模型F1 score
            f1_score = metrics.f1_score(y_true=np.array(true_y), y_pred=np.array(test_predictions), average='weighted')
            # 详细指标报告  Precision， Recall， F1
            report = metrics.classification_report(y_true=np.array(true_y), y_pred=np.array(test_predictions))

            logging.info('Test Epoch: %d , Loss: %.6f , Acc: %.6f , F1 Socre: %.6f' % (epoch,
                                                                                       float(np.mean(np.array(test_loss_list))),
                                                                                       float(np.mean(np.array(test_accuracy_list))),
                                                                                       f1_score))
            print('Report: \n', report)
            # 当前epoch产生的模型F1值超过最好指标时，保存当前模型
            if best_f1_score < f1_score:
                best_f1_score = f1_score
                saver.save(sess=self.sess, save_path=FastTextConfig.MODEL_SAVE_PATH)
                logging.info('Save Model Success ...')

    def predict_all(self):
        # 实例化并加载模型
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=FastTextConfig.MODEL_SAVE_PATH)
        # 加载词汇->ID映射表
        # self.word_to_id = read_vocab(FastTextConfig.VOCAB_PATH)
        # _, id_to_label = read_label_db()
        # 从数据库获取所有数据并预测
        data = ClassifyData.objects.all()
        for d in data:
            sentence_id = d['pre_text']
            # 对句子预处理并进行ID表示
            # sentence_id = self.pre_process(sentence)
            feed_dict = {self.input_x: [sentence_id], self.input_keep_prob: 1.0}
            predict = self.sess.run(self.output, feed_dict=feed_dict)[0]
            # label = id_to_label.get(predict)
            data.update(predict_tag=predict)


    # def predict(self, sentence):
    #     # 实例化并加载模型
    #     saver = tf.train.Saver()
    #     saver.restore(sess=self.sess, save_path=FastTextConfig.MODEL_SAVE_PATH)
    #     # 加载词汇->ID映射表
    #     self.word_to_id = read_vocab(FastTextConfig.VOCAB_PATH)
    #
    #     # 对句子预处理并进行ID表示
    #     sentence_id = self.pre_process(sentence)
    #     feed_dict = {self.input_x: [sentence_id], self.input_keep_prob: 1.0}
    #     predict = self.sess.run(self.output, feed_dict=feed_dict)[0]
    #
    #     return predict

    def pre_process(self, sentence):
        '''
        文本数据预处理
        :param sentence: 输入的文本句子
        :return:
        '''
        # 分词，去除停用词
        sentence_seg = [word for word in text_processing(sentence).split(' ') if word not in self.stopwords and not word.isdigit()]
        # 将词汇映射为ID
        sentence_id = []
        for word in sentence_seg:
            if word in self.word_to_id:
                sentence_id.append(self.word_to_id[word])
            else:
                sentence_id.append(self.word_to_id['<UNK>'])
        # 对文本长度进行padding填充
        sentence_length = len(sentence_id)
        if sentence_length > FastTextConfig.SEQ_LENGTH:
            sentence_id = sentence_id[: FastTextConfig.SEQ_LENGTH]
        else:
            sentence_id.extend([self.word_to_id['<PAD>']] * (FastTextConfig.SEQ_LENGTH - sentence_length))

        return sentence_id
