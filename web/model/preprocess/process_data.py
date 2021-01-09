import logging
from tqdm import tqdm
import os
import jieba
from collections import Counter
import numpy as np

from data.models import ClassifyTag
from model.config.model_config import STOP_WORDS_PATH, FastTextConfig


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
    stopwords = [word.replace('\n', '').strip() for word in open(STOP_WORDS_PATH, encoding='UTF-8')]
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


def data_transform(input_data, input_label, word_to_id, label_to_id):
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
    indices = np.random.permutation(np.arange(len(sentence_id)))
    datas = np.array(sentence_id)[indices]
    labels = np.array(label_id)[indices]

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
