import os
import django
import sys

# 这两行很重要，用来寻找项目根目录，os.path.dirname要写多少个根据要运行的python文件到根目录的层数决定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()


from algo.model.model_config import FastTextConfig, BertBilstmCrfConfig
from algo.model.process_data import pre_process, load_dataset, build_vocab, read_vocab, build_label, read_label, \
    data_transform, load_raw_data, DataProcess
from algo.models import ClassifyData, NerData, TripleExtractData
from algo.model.bert_bilstm_crf import train_sample
from algo.model.bert_relation_extraction import data_generator, train_data, batch_size, \
    BertTripleExtract, load_data
from algo.model.msra_preprocessing import read_file


'''
if __name__ == "__main__":
    """
    分类数据
    将原始文本和id化后的序列存储
    """
    # 数据集预处理
    if not os.path.exists(FastTextConfig.PREPROCESS_PATH):
        pre_process(FastTextConfig.ORIGINAL_DATA_PATH, FastTextConfig.PREPROCESS_PATH)
    sentences, labels = load_dataset(FastTextConfig.PREPROCESS_PATH)      # 加载数据集
    # 构建词汇映射表
    if not os.path.exists(FastTextConfig.VOCAB_PATH):
        build_vocab(sentences, FastTextConfig.VOCAB_PATH)
    word_to_id = read_vocab(FastTextConfig.VOCAB_PATH)      # 读取词汇表及其映射关系
    # 构建类别映射表
    if not os.path.exists(FastTextConfig.LABEL_PATH):
        build_label(labels, FastTextConfig.LABEL_PATH)
    label_to_id = read_label(FastTextConfig.LABEL_PATH)     # 读取类别表及其映射关系

    # 构建训练数据集
    data_sentences, data_labels = data_transform(sentences, labels, word_to_id, label_to_id)

    # 获得原始数据和标签
    raw_sentences, _ = load_raw_data(FastTextConfig.ORIGINAL_DATA_PATH)

    # 将数据存储到数据库
    data = ClassifyData.objects.all()
    data.delete()
    for sentence, id_sentence in zip(raw_sentences, data_sentences):
        model = ClassifyData(text=sentence, pre_text=",".join([str(i) for i in id_sentence]))
        model.save()
'''

"""
if __name__ == "__main__":
    '''
    # 训练bert-bilstm-crf
    '''
    # columns = ['model_name','epoch', 'loss', 'acc', 'val_loss', 'val_acc', 'f1', 'recall']
    info_list = train_sample(epochs=1)
    for info in info_list:
        print(info)
"""

"""
if __name__ == "__main__":
    '''
    将实体识别数据存储到数据库
    '''
    path_train1 = os.path.join(BertBilstmCrfConfig.MSRA_DIR, "data.txt")
    texts = read_file(path_train1)
    sentences = []
    for t in texts:
        words = t.split(' ')
        sentence = [w.split('/')[0] for w in words]
        sentences.append(sentence)
    data = NerData.objects.all()
    data.delete()
    for s in sentences:
        model = NerData(text="".join(s))
        model.save()
"""


"""
if __name__ == '__main__':
    model = BertTripleExtract()
    model.build()
    model.train()
    model.predict_all()
"""

if __name__ == "__main__":
    """
    三元组数据存储到数据库
    """
    train_data = load_data('/Users/wanli/create/project/CatchingFire/backend/data/triple/train_data_small')
    for t in train_data:
        data = TripleExtractData(text=t['text'])
        data.save()

