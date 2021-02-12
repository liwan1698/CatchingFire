import os
import django
import sys

# 这两行很重要，用来寻找项目根目录，os.path.dirname要写多少个根据要运行的python文件到根目录的层数决定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()


from algo.model.model_config import FastTextConfig
from algo.model.process_data import pre_process, load_dataset, build_vocab, read_vocab, build_label, read_label, \
    data_transform, load_raw_data
from algo.models import ClassifyData


if __name__ == "__main__":
    """
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

