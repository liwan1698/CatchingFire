import logging
from tqdm import tqdm
import os
import jieba
from collections import Counter
import numpy as np
from algo.model.model_config import BertBilstmCrfConfig
from algo.model.msra_preprocessing import msra_preprocessing
from algo.models import ClassifyTag
from algo.model.model_config import FastTextConfig


def load_raw_data(data_path):
    '''
    读取分类原始数据
    :param data_path:
    :return:
    '''
    labels, sentences = [], []
    for line in tqdm(open(data_path, encoding='UTF-8')):
        label_sentence = str(line).strip().replace('\n', '').split('\t')    # 去除收尾空格、结尾换行符\n、使用\t切分
        label = label_sentence[0]
        sentence = label_sentence[1].replace('\t', '').replace('\n', '').replace(' ', '')   # 符号过滤
        labels.append(label)
        sentences.append(sentence)
    return sentences, labels


def pre_process(data_path, preprocess_path):
    '''
    原始数据预处理
    :param data_path: 原始文本文件路径
    :param preprocess_path: 预处理后的数据存储路径
    :return:
    '''
    # 加载停用词表
    logging.info('Start Preprocess ...')
    preprocess_file = open(preprocess_path, mode='w', encoding='UTF-8')
    # 加载停用词表
    stopwords = [word.replace('\n', '').strip() for word in open(FastTextConfig.STOP_WORDS_PATH, encoding='UTF-8')]
    for line in tqdm(open(data_path, encoding='UTF-8')):
        label_sentence = str(line).strip().replace('\n', '').split('\t')    # 去除收尾空格、结尾换行符\n、使用\t切分
        label = label_sentence[0]
        sentence = label_sentence[1].replace('\t', '').replace('\n', '').replace(' ', '')   # 符号过滤
        sentence = [word for word in text_processing(sentence).split(' ') if word not in stopwords and not word.isdigit()]
        preprocess_file.write(label + '\t' + ' '.join(sentence) + '\n')

    preprocess_file.close()

def text_processing(text):
    '''
    文本数据预处理，分词，去除停用词
    :param text: 文本数据sentence
    :return: 以空格为分隔符进行分词/分字结果
    '''
    # 删除（）里的内容
    # text = re.sub('（[^（.]*）', '', text)
    # 只保留中文部分
    text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
    # 利用jieba进行分词
    words = list(jieba.cut(text))
    # 不分词
    # words = [x for x in ''.join(text)]
    return ' '.join(words)


def load_dataset(data_path):
    '''
    从本地磁盘加载经过预处理的数据集，避免每次都进行预处理操作
    :param data_path: 预处理好的数据集路径
    :return: 句子列表，标签列表
    '''
    sentences = []
    labels = []
    # 加载停用词表
    logging.info('Load Dataset ...')
    for line in tqdm(open(data_path, encoding='UTF-8')):
        try:
            label_sentence = str(line).strip().replace('\n', '').split('\t')
            label = label_sentence[0]                           # 标签
            sentence = label_sentence[1]                        # 以空格为切分的content
            sentence = [word for word in sentence.split(' ')]
            sentences.append(sentence)
            labels.append(label)
        except:
            logging.info('Load Data Error ... msg: {}'.format(line))    # 部分数据去除英文和数字后为空，跳过异常
            continue

    return sentences, labels


def build_vocab(input_data, vocab_path):
    '''
    根据数据集构建词汇表，存储到本地备用
    :param input_data: 全部句子集合 [n] n为数据条数
    :param vocab_path: 词表文件存储路径
    :return:
    '''
    logging.info('Build Vocab ...')
    all_data = []       # 全部句子集合
    for content in input_data:
        all_data.extend(content)

    counter = Counter(all_data)     # 词频统计
    count_pairs = counter.most_common(FastTextConfig.VOCAB_SIZE - 2)    # 对词汇按次数进行降序排序
    words, _ = list(zip(*count_pairs))              # 将(word, count)元祖形式解压，转换为列表list
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<UNK>'] + list(words)  # 增加一个OOV标识的编码
    words = ['<PAD>'] + list(words)  # 增加一个PAD标识的编码
    open(vocab_path, mode='w', encoding='UTF-8').write('\n'.join(words) + '\n')


def read_vocab(vocab_path):
    """
    读取词汇表，构建 词汇-->ID 映射字典
    :param vocab_path: 词表文件路径
    :return: 词表，word_to_id
    """
    words = [word.replace('\n', '').strip() for word in open(vocab_path, encoding='UTF-8')]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def build_label(input_label, label_path):
    '''
    根据标签集构建标签表，存储到本地备用
    :param input_label: 全部标签集合
    :param label_path: 标签文件存储路径
    :return:
    '''
    logging.info('Build Label ...')
    all_label = set(input_label)
    open(label_path, mode='w', encoding='UTF-8').write('\n'.join(all_label))


def read_label(label_path):
    '''
    读取类别表，构建 类别-->ID 映射字典
    :param label_path: 类别文件路径
    :return: 类别表，label_to_id
    '''
    labels = [label.replace('\n', '').strip() for label in open(label_path, encoding='UTF-8')]
    label_to_id = dict(zip(labels, range(len(labels))))

    return label_to_id


def read_label_db():
    '''
    读取类别表，构建 类别-->ID 映射字典
    :return: 类别表，label_to_id
    '''
    tags = ClassifyTag.objects.values("tag")
    labels = [t["tag"] for t in tags]
    label_to_id = dict(zip(labels, range(len(labels))))
    id_to_label = dict(zip(range(len(labels)), labels))
    return label_to_id, id_to_label


def data_transform(input_data, input_label, word_to_id, label_to_id, shuffle=False):
    '''
    数据预处理，将文本和标签映射为ID形式
    :param input_data: 文本数据集合
    :param input_label: 标签集合
    :param word_to_id: 词汇——ID映射表
    :param label_to_id: 标签——ID映射表
    :return: ID形式的文本，ID形式的标签
    '''
    logging.info('Sentence Trans To ID ...')
    sentence_id = []
    # 将文本转换为ID表示[1,6,2,3,5,8,9,4]
    for sentence in tqdm(input_data):
        sentence_temp = []
        for word in sentence:
            if word in word_to_id:
                sentence_temp.append(word_to_id[word])
            else:
                sentence_temp.append(word_to_id['<UNK>'])

        # 对文本长度进行padding填充
        sentence_length = len(sentence_temp)
        if sentence_length > FastTextConfig.SEQ_LENGTH:
            sentence_temp = sentence_temp[: FastTextConfig.SEQ_LENGTH]
        else:
            sentence_temp.extend([word_to_id['<PAD>']] * (FastTextConfig.SEQ_LENGTH - sentence_length))
        sentence_id.append(sentence_temp)


    # 将标签转换为ID形式
    logging.info('Label Trans To One-Hot ...')
    label_id = []
    for label in tqdm(input_label):
        label_id_temp = np.zeros([FastTextConfig.NUM_CLASSES])
        if label in label_to_id:
            label_id_temp[label_to_id[label]] = 1
            label_id.append(label_id_temp)

    # shuffle
    datas = np.array(sentence_id)
    labels = np.array(label_id)
    if shuffle:
        indices = np.random.permutation(np.arange(len(sentence_id)))
        datas = datas[indices]
        labels = labels[indices]
    return datas, labels


def creat_batch_data(input_data, input_label, batch_size):
    '''
    将数据集以batch_size大小进行切分
    :param input_data: 数据列表
    :param input_label: 标签列表
    :param batch_size: 批大小
    :return:
    '''
    max_length = len(input_data)            # 数据量
    max_index = max_length // batch_size    # 最大批次
    # shuffle
    indices = np.random.permutation(np.arange(max_length))
    data_shuffle = np.array(input_data)[indices]
    label_shuffle = np.array(input_label)[indices]

    batch_data, batch_label = [], []
    for index in range(max_index):
        start = index * batch_size                              # 起始索引
        end = min((index + 1) * batch_size, max_length)         # 结束索引，可能为start + batch_size 或max_length
        batch_data.append(data_shuffle[start: end])
        batch_label.append(label_shuffle[start: end])

        if (index + 1) * batch_size > max_length:               # 如果结束索引超过了数据量，则结束
            break

    return batch_data, batch_label


class DataProcess(object):
    def __init__(self,
                 max_len=100,
                 data_type='data',  # 'data', 'data2', 'msra', 'renmin'
                 model='other',  # 'other'、'bert' bert 数据处理需要单独进行处理
                 ):
        """
        数据处理
        :param max_len: 句子最长的长度，默认为保留100
        :param data_type: 数据类型，当前支持四种数据类型
        """
        self.w2i = get_w2i()  # word to index
        self.tag2index = get_tag2index()  # tag to index
        self.vocab_size = len(self.w2i)
        self.tag_size = len(self.tag2index)
        self.unk_flag = unk_flag
        self.pad_flag = pad_flag
        self.max_len = max_len
        self.model = model

        self.unk_index = self.w2i.get(unk_flag, 101)
        self.pad_index = self.w2i.get(pad_flag, 1)
        self.cls_index = self.w2i.get(cls_flag, 102)
        self.sep_index = self.w2i.get(sep_flag, 103)

        if data_type == 'msra':
            self.base_dir = BertBilstmCrfConfig.MSRA_DIR
            msra_preprocessing()
        else:
            raise RuntimeError('type must be "data", "msra", "renmin" or "data2"')

    def get_data(self, one_hot: bool = True) -> ([], [], [], []):
        """
        获取数据，包括训练、测试数据中的数据和标签
        :param one_hot:
        :return:
        """
        # 拼接地址
        path_train = os.path.join(self.base_dir, "train_small.txt")
        path_test = os.path.join(self.base_dir, "test_small.txt")

        # 读取数据
        if self.model == 'bert':
            train_data, train_label = self.__bert_text_to_index(path_train)
            test_data, test_label = self.__bert_text_to_index(path_test)
        else:
            train_data, train_label = self.__text_to_indexs(path_train)
            test_data, test_label = self.__text_to_indexs(path_test)

        # 进行 one-hot处理
        if one_hot:
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
            train_label = label_to_one_hot(index=train_label)
            test_label = label_to_one_hot(index=test_label)
        else:
            train_label = np.expand_dims(train_label, 2)
            test_label = np.expand_dims(test_label, 2)
        return train_data, train_label, test_data, test_label

    def num2tag(self):
        return dict(zip(self.tag2index.values(), self.tag2index.keys()))

    def i2w(self):
        return dict(zip(self.w2i.values(), self.w2i.keys()))

    # texts 转化为 index序列
    def __text_to_indexs(self, file_path: str) -> ([], []):
        data, label = [], []
        with open(file_path, 'r') as f:
            line_data,  line_label = [], []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    char_index = self.w2i.get(w, self.w2i[self.unk_flag])
                    tag_index = self.tag2index.get(t, 0)
                    line_data.append(char_index)
                    line_label.append(tag_index)
                else:
                    if len(line_data) < self.max_len:
                        pad_num = self.max_len - len(line_data)
                        line_data = [self.pad_index]*pad_num + line_data
                        line_label = [0]*pad_num + line_label
                    else:
                        line_data = line_data[:self.max_len]
                        line_label = line_label[:self.max_len]
                    data.append(line_data)
                    label.append(line_label)
                    line_data, line_label = [], []
        return np.array(data), np.array(label)

    def __bert_text_to_index(self, file_path: str):
        """
        bert的数据处理
        处理流程 所有句子开始添加 [CLS] 结束添加 [SEP]
        bert需要输入 ids和types所以需要两个同时输出
        由于我们句子都是单句的，所以所有types都填充0

        :param file_path:  文件路径
        :return: [ids, types], label_ids
        """
        data_ids = []
        data_types = []
        label_ids = []
        with open(file_path, 'r') as f:
            line_data_ids = []
            line_data_types = []
            line_label = []
            for line in f:
                if line != '\n':
                    w, t = line.split()
                    # bert 需要输入index和types 由于我们这边都是只有一句的，所以type都为0
                    w_index = self.w2i.get(w, self.unk_index)
                    t_index = self.tag2index.get(t, 0)
                    line_data_ids.append(w_index)  # index
                    line_data_types.append(0)  # types
                    line_label.append(t_index)  # label index
                else:
                    # 处理填充开始和结尾 bert 输入语句每个开始需要填充[CLS] 结束[SEP]
                    max_len_buff = self.max_len-2
                    if len(line_data_ids) > max_len_buff: # 先进行截断
                        line_data_ids = line_data_ids[:max_len_buff]
                        line_data_types = line_data_types[:max_len_buff]
                        line_label = line_label[:max_len_buff]
                    line_data_ids = [self.cls_index] + line_data_ids + [self.sep_index]
                    line_data_types = [0] + line_data_types + [0]
                    line_label = [0] + line_label + [0]

                    # padding
                    if len(line_data_ids) < self.max_len: # 填充到最大长度
                        pad_num = self.max_len - len(line_data_ids)
                        line_data_ids = [self.pad_index]*pad_num + line_data_ids
                        line_data_types = [0] * pad_num + line_data_types
                        line_label = [0] * pad_num + line_label
                    data_ids.append(np.array(line_data_ids))
                    data_types.append(np.array(line_data_types))
                    label_ids.append(np.array(line_label))
                    line_data_ids = []
                    line_data_types = []
                    line_label = []
        return [np.array(data_ids), np.array(data_types)], np.array(label_ids)


# 获取词典
unk_flag = '[UNK]'
pad_flag = '[PAD]'
cls_flag = '[CLS]'
sep_flag = '[SEP]'


# 获取 word to index 词典
def get_w2i(vocab_path = BertBilstmCrfConfig.VOCAB_PATH):
    w2i = {}
    with open(vocab_path, 'r') as f:
        while True:
            text = f.readline()
            if not text:
                break
            text = text.strip()
            if text and len(text) > 0:
                w2i[text] = len(w2i) + 1
    return w2i


# 获取 tag to index 词典
def get_tag2index():
    return {"O": 0,
            "B-PER": 1, "I-PER": 2,
            "B-LOC": 3, "I-LOC": 4,
            "B-ORG": 5, "I-ORG": 6
            }


if __name__ == '__main__':
    get_w2i()


if __name__ == '__main__':

    # dp = DataProcess(data_type='data')
    # x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    #
    # print(y_train[:1, :1, :100])

    dp = DataProcess(data_type='data', model='bert')
    x_train, y_train, x_test, y_test = dp.get_data(one_hot=True)
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(y_train.shape)
    print(x_test[0].shape)
    print(x_test[1].shape)
    print(y_test.shape)

    print(y_train[:1, :1, :100])

    pass



